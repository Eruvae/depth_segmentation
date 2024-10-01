#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <message_filters/synchronizer.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>

#ifdef MASKRCNNROS_AVAILABLE
#include <mask_rcnn_ros/Result.h>
#endif

#include "depth_segmentation/depth_segmentation.h"
#include "depth_segmentation/ros_common.h"

std::vector<depth_segmentation::OverlapSegment> default_segment;

struct PointSurfelLabel {
  PCL_ADD_POINT4D;
  PCL_ADD_NORMAL4D;
  PCL_ADD_RGB;
  uint32_t instance_label;
  uint8_t semantic_label;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointSurfelLabel,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z)(float, rgb, rgb)(
        uint32_t, instance_label, instance_label)(uint8_t, semantic_label,
                                                 semantic_label))

class DepthSegmentationNode {
 public:
  DepthSegmentationNode()
      : node_handle_("~"),
        image_transport_(node_handle_),
        camera_info_ready_(false),
        depth_camera_(),
        rgb_camera_(),
        params_(),
        camera_tracker_(depth_camera_, rgb_camera_),
        depth_segmenter_(depth_camera_, params_) {
    node_handle_.param<bool>("semantic_instance_segmentation/enable",
                             params_.semantic_instance_segmentation.enable,
                             params_.semantic_instance_segmentation.enable);
    node_handle_.param<float>(
        "semantic_instance_segmentation/overlap_threshold",
        params_.semantic_instance_segmentation.overlap_threshold,
        params_.semantic_instance_segmentation.overlap_threshold);

    node_handle_.param<std::string>("depth_image_sub_topic", depth_image_topic_,
                                    depth_segmentation::kDepthImageTopic);
    node_handle_.param<std::string>("rgb_image_sub_topic", rgb_image_topic_,
                                    depth_segmentation::kRgbImageTopic);
    node_handle_.param<std::string>("depth_camera_info_sub_topic",
                                    depth_camera_info_topic_,
                                    depth_segmentation::kDepthCameraInfoTopic);
    node_handle_.param<std::string>("rgb_camera_info_sub_topic",
                                    rgb_camera_info_topic_,
                                    depth_segmentation::kRgbCameraInfoTopic);
    node_handle_.param<std::string>(
        "semantic_instance_segmentation_sub_topic",
        semantic_instance_segmentation_topic_,
        depth_segmentation::kSemanticInstanceSegmentationTopic);

    ROS_INFO_STREAM("Semantic instance segmentation topic: "<<semantic_instance_segmentation_topic_);
    node_handle_.param<std::string>("world_frame", world_frame_,
                                    depth_segmentation::kTfWorldFrame);
    node_handle_.param<std::string>("camera_frame", camera_frame_,
                                    depth_segmentation::kTfDepthCameraFrame);

    node_handle_.param<std::string>("pointcloud_sub_topic", pc2_msg_topic_,
                                    depth_segmentation::kPointCloud2Topic);

    node_handle_.param<std::string>("joint_states_topic", joint_states_topic_,
                                    depth_segmentation::kJointStatesTopic);

    node_handle_.param<bool>("forward_labeled_segments_only", forward_labeled_segments_only_,
                                    depth_segmentation::kForwardLabeledSegmentsOnly);  
    node_handle_.param<bool>("use_overlap_bits_only", use_overlap_bits_only_,
                                    depth_segmentation::kUSeOverlapBitsOnly);
    node_handle_.param<bool>("publish_while_moving", publish_while_moving_,
                                    depth_segmentation::kPublishWhileMoving);
    node_handle_.param<bool>("use_joint_velocities", use_joint_velocities_,
                                    depth_segmentation::kUseJointVelocities);
    node_handle_.param<bool>("use_selective_joints", use_selective_joints_,
                                    depth_segmentation::kUseSelectiveJoints);
    node_handle_.param<std::vector<std::string>>("selective_joint_names", selective_joint_names_,
                                    depth_segmentation::kSelectiveJointNames);
    node_handle_.param<float>("max_joint_difference", max_joint_difference_,
                                    depth_segmentation::kMaxJointDifference);
    node_handle_.param<float>("max_joint_velocity", max_joint_velocity_,
                                    depth_segmentation::kMaxJointVelocity);
    node_handle_.param<float>("wait_time_stationary_", wait_time_stationary_,
                                    depth_segmentation::kWaitTimeAfterStationary);
    node_handle_.param<bool>("use_stability_score", use_stability_score_,
                                    depth_segmentation::kUSeStabilityScores);

    node_handle_.param<int>("min_segment_size", min_segment_size_,
                                    depth_segmentation::kMinSegmentSize);                                
    node_handle_.param<float>("min_segment_depth", min_segment_depth_,
                                    depth_segmentation::kMinSegmentDepth);                           
    node_handle_.param<float>("max_segment_depth", max_segment_depth_,
                                    depth_segmentation::kMaxSegmentDepth);

    node_handle_.param<bool>("publish_scene_as_xyzl", publish_scene_as_xyzl_, false);

    depth_image_sub_ = new image_transport::SubscriberFilter(
        image_transport_, depth_image_topic_, 1);
    rgb_image_sub_ = new image_transport::SubscriberFilter(image_transport_,
                                                           rgb_image_topic_, 1);
    depth_info_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo>(
        node_handle_, depth_camera_info_topic_, 1);
    rgb_info_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo>(
        node_handle_, rgb_camera_info_topic_, 1);

    pc2_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(
        node_handle_, pc2_msg_topic_, 1);

    joint_states_sub_ = node_handle_.subscribe(joint_states_topic_, 1, &DepthSegmentationNode::registerJointState, this);
    constexpr int kQueueSize = 100;

#ifndef MASKRCNNROS_AVAILABLE
    if (params_.semantic_instance_segmentation.enable) {
      params_.semantic_instance_segmentation.enable = false;
      ROS_WARN_STREAM(
          "Turning off semantic instance segmentation "
          "as mask_rcnn_ros is disabled.");
    }
#endif

    if (params_.semantic_instance_segmentation.enable) {
#ifdef MASKRCNNROS_AVAILABLE
      instance_segmentation_sub_ =
          new message_filters::Subscriber<mask_rcnn_ros::Result>(
              node_handle_, semantic_instance_segmentation_topic_, 1);

      image_segmentation_sync_policy_ =
          new message_filters::Synchronizer<ImageSegmentationSyncPolicy>(
              ImageSegmentationSyncPolicy(kQueueSize), *depth_image_sub_,
              *rgb_image_sub_, *instance_segmentation_sub_); //, *pc2_sub_);

      image_segmentation_sync_policy_->registerCallback(boost::bind(
          &DepthSegmentationNode::imageSegmentationCallback, this, _1, _2, _3)); //, _4));
#endif
    } else {
      image_sync_policy_ = new message_filters::Synchronizer<ImageSyncPolicy>(
          ImageSyncPolicy(kQueueSize), *depth_image_sub_, *rgb_image_sub_);

      image_sync_policy_->registerCallback(
          boost::bind(&DepthSegmentationNode::imageCallback, this, _1, _2));
    }

    camera_info_sync_policy_ =
        new message_filters::Synchronizer<CameraInfoSyncPolicy>(
            CameraInfoSyncPolicy(kQueueSize), *depth_info_sub_, *rgb_info_sub_);

    camera_info_sync_policy_->registerCallback(
        boost::bind(&DepthSegmentationNode::cameraInfoCallback, this, _1, _2));

    point_cloud2_segment_pub_ =
        node_handle_.advertise<sensor_msgs::PointCloud2>("/depth_segmentation_node/object_segment",
                                                         1000);
    point_cloud2_scene_pub_ =
        node_handle_.advertise<sensor_msgs::PointCloud2>("segmented_scene", 1);

    node_handle_.param<bool>("visualize_segmented_scene",
                             params_.visualize_segmented_scene,
                             params_.visualize_segmented_scene);

    tf_buffer_.reset(new tf2_ros::Buffer(ros::Duration(tf2::BufferCore::DEFAULT_CACHE_TIME)));
    tf_listener_.reset(new tf2_ros::TransformListener(*tf_buffer_, node_handle_));
    last_robot_moved_time_ = ros::Time::now();
    ROS_INFO("Constructor");
  }

 private:
  ros::NodeHandle node_handle_;
  image_transport::ImageTransport image_transport_;
  tf::TransformBroadcaster transform_broadcaster_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      ImageSyncPolicy;

#ifdef MASKRCNNROS_AVAILABLE
  typedef message_filters::sync_policies::ExactTime<
      sensor_msgs::Image, sensor_msgs::Image, mask_rcnn_ros::Result> //, sensor_msgs::PointCloud2>
      ImageSegmentationSyncPolicy;
#endif

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>
      CameraInfoSyncPolicy;

  bool camera_info_ready_;
  depth_segmentation::DepthCamera depth_camera_;
  depth_segmentation::RgbCamera rgb_camera_;

  depth_segmentation::Params params_;

 public:
  depth_segmentation::CameraTracker camera_tracker_;
  depth_segmentation::DepthSegmenter depth_segmenter_;

 private:
  std::string rgb_image_topic_;
  std::string rgb_camera_info_topic_;
  std::string depth_image_topic_;
  std::string depth_camera_info_topic_;
  std::string semantic_instance_segmentation_topic_;
  std::string pc2_msg_topic_;
  std::string joint_states_topic_;
  std::string world_frame_;
  std::string camera_frame_;

  bool forward_labeled_segments_only_;
  bool use_overlap_bits_only_;

  image_transport::SubscriberFilter* depth_image_sub_;
  image_transport::SubscriberFilter* rgb_image_sub_;

  message_filters::Subscriber<sensor_msgs::PointCloud2>* pc2_sub_;
  ros::Subscriber joint_states_sub_;

  message_filters::Subscriber<sensor_msgs::CameraInfo>* depth_info_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo>* rgb_info_sub_;

  ros::Publisher point_cloud2_segment_pub_;
  ros::Publisher point_cloud2_scene_pub_;

  sensor_msgs::JointState joint_state_, prev_joint_state_;
  bool joint_state_recd_ = false;  
  ros::Time last_robot_moved_time_;
  ros::Time last_robot_steady_time_;

  message_filters::Synchronizer<ImageSyncPolicy>* image_sync_policy_;

  message_filters::Synchronizer<CameraInfoSyncPolicy>* camera_info_sync_policy_;

  bool publish_while_moving_;
  bool use_joint_velocities_;
  bool use_selective_joints_;
  std::vector<std::string> selective_joint_names_;
  float max_joint_velocity_;
  float max_joint_difference_;
  float wait_time_stationary_;
  bool use_stability_score_;
  bool use_transform_;
  int min_segment_size_;
  float min_segment_depth_;
  float max_segment_depth_;

  bool stable_depth_image_available_ = false;
  bool robot_moving_ = true;
  bool robot_steady_ = false;
  bool vecs_initialized_ = false;

  bool publish_scene_as_xyzl_ = false;

#ifdef MASKRCNNROS_AVAILABLE
  message_filters::Subscriber<mask_rcnn_ros::Result>*
      instance_segmentation_sub_;
  message_filters::Synchronizer<ImageSegmentationSyncPolicy>*
      image_segmentation_sync_policy_;
#endif

  void publish_tf(const cv::Mat cv_transform, const ros::Time& timestamp) {
    // Rotate such that the world frame initially aligns with the camera_link
    // frame.
    static const cv::Mat kWorldAlign =
        (cv::Mat_<double>(4, 4) << 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cv_transform_world_aligned = cv_transform * kWorldAlign;

    tf::Vector3 translation_tf(cv_transform_world_aligned.at<double>(0, 3),
                               cv_transform_world_aligned.at<double>(1, 3),
                               cv_transform_world_aligned.at<double>(2, 3));

    tf::Matrix3x3 rotation_tf;
    for (size_t i = 0u; i < 3u; ++i) {
      for (size_t j = 0u; j < 3u; ++j) {
        rotation_tf[j][i] = cv_transform_world_aligned.at<double>(j, i);
      }
    }
    tf::Transform transform;
    transform.setOrigin(translation_tf);
    transform.setBasis(rotation_tf);

    transform_broadcaster_.sendTransform(tf::StampedTransform(
        transform, timestamp, camera_frame_, world_frame_));
  }

  void fillPoint(const cv::Vec3f& point, const cv::Vec3f& normals,
                 const cv::Vec3f& colors, pcl::PointSurfel* point_pcl) {
    point_pcl->x = point[0];
    point_pcl->y = point[1];
    point_pcl->z = point[2];
    point_pcl->normal_x = normals[0];
    point_pcl->normal_y = normals[1];
    point_pcl->normal_z = normals[2];
    if(colors[0] > 0 && colors[0] < 256 && colors[1] > 0 && colors[1] < 256 && colors[2] > 0 && colors[2] < 256)
    {
      point_pcl->r = colors[0];
      point_pcl->g = colors[1];
      point_pcl->b = colors[2];
    }
    else
    {
      point_pcl->r = 255; //colors[0];
      point_pcl->g = 0; //colors[1];
      point_pcl->b = 0; //colors[2];
    }
  }

  void fillPoint(const cv::Vec3f& point, const cv::Vec3f& normals,
                 const cv::Vec3f& colors, const size_t& semantic_label,
                 const uint32_t& instance_label, PointSurfelLabel* point_pcl) {
    point_pcl->x = point[0];
    point_pcl->y = point[1];
    point_pcl->z = point[2];
    point_pcl->normal_x = normals[0];
    point_pcl->normal_y = normals[1];
    point_pcl->normal_z = normals[2];
    if(colors[0] > 0 && colors[0] < 256 && colors[1] > 0 && colors[1] < 256 && colors[2] > 0 && colors[2] < 256)
    {
      point_pcl->r = colors[0];
      point_pcl->g = colors[1];
      point_pcl->b = colors[2];
    }
    else
    {
      point_pcl->r = 255; //colors[0];
      point_pcl->g = 0; //colors[1];
      point_pcl->b = 0; //colors[2];
    }

    point_pcl->semantic_label = semantic_label;
    point_pcl->instance_label = instance_label;

    //ROS_WARN_STREAM_THROTTLE(1, "New labeled point with semantic label:  "<<int16_t(point_pcl->semantic_label)<<"& instance label: "<<int16_t(point_pcl->instance_label));
  }

  /*void convertDepthSegement2InstancePclSegment(const depth_segmentation::OverlapSegment& overlap_segment, const std_msgs::Header& header, pcl::PointCloud<PointSurfelLabel>::Ptr scene_pcl, bool use_overlap_mask)
  {
    CHECK_GT(overlap_segment.segment.points.size(), 0u);
    pcl::PointCloud<PointSurfelLabel>::Ptr segment_pcl(
            new pcl::PointCloud<PointSurfelLabel>);
        
    sensor_msgs::PointCloud2 pcl2_msg;
    std::size_t num_pts = 0u; 
    for (std::size_t i = 0u; i < overlap_segment.segment.points.size(); ++i) {
      PointSurfelLabel point_pcl;
      uint8_t semantic_label = 0u;
      uint8_t instance_label = 0u;
      if (segment.instance_label.size() > 0u) {
        instance_label = *(overlap_segment.segment.instance_label.begin());
        semantic_label = *(overlap_segment.segment.semantic_label.begin());
      }
      if (overlap_segment.segment.instance_label.size() > 0u)
      {
        if(num_pts < 1u)
        {
          num_labelled_segments++;
          ROS_WARN_STREAM("New labeled segment of size "<<overlap_segment.segment.points.size()<<"found  with semantic label:  "<<semantic_label<<"& instance label: "<<instance_label);
        }
        //ROS_WARN_STREAM_THROTTLE(5, "Instance label: "<<instance_label<<"\tSemantic label: "<<semantic_label);
        if(use_overlap_mask)
        {
          if(overlap_segment)
        }
        num_pts++;
        fillPoint(segment.points[i], segment.normals[i],
                  segment.original_colors[i], semantic_label, instance_label,
                  &point_pcl);
        segment_pcl->push_back(point_pcl);
        scene_pcl->push_back(point_pcl);
      }
    }
    ROS_DEBUG_STREAM("\tInserted "<<num_pts<< "points for above segment");
    if(segment_pcl->points.size()> 0)
    {
      sensor_msgs::PointCloud2 pcl2_msg;
      pcl::toROSMsg(*segment_pcl, pcl2_msg);
      pcl2_msg.header.stamp = header.stamp;
      pcl2_msg.header.frame_id = header.frame_id;
      point_cloud2_segment_pub_.publish(pcl2_msg);
    }
  }*/

  void registerJointState(const sensor_msgs::JointState& joint_state)
  {
    if((joint_state.header.stamp - joint_state_.header.stamp).toSec() < 0.05)
      return;

    // sensor_msgs::JointState head_state;
    // head_state.header = joint_state.header;
    // for(int i = 0; i < joint_state.name.size(); i++)
    // {
    //   if(joint_state.name[i].find("head_joint") != std::string::npos)
    //   {
    //     head_state.name.push_back(joint_state.name[i]);
    //     head_state.position.push_back(joint_state.position[i]);
    //   }
    // }
    // if(head_state.name.size()< 6)
    // {
    //   if(head_state.name.size() > 0)
    //     ROS_WARN("Full head state not recd");
    //   return;
    // }

    if(joint_state_recd_) 
    {
      prev_joint_state_ = joint_state_;
    }
    else
    {
      prev_joint_state_ = joint_state; //head_state;
      joint_state_recd_ = true;
    }
    joint_state_ = joint_state; //head_state;
  }

  bool is_segment_depth_valid(const depth_segmentation::Segment& segment)
  {
    int segment_size = segment.points.size();
    if(segment.points.size() < min_segment_size_)
    {
      ROS_INFO_STREAM("Segment has too few points. Hence not publishing");
      return false;
    }
    float avg_depth =     0.0f;
    int points_greater_depth = 0;
    int points_lesser_depth = 0;
    for(const auto& point: segment.points)
    {
      if(point[2] < min_segment_depth_)
        points_lesser_depth++;

      if(point[2] > max_segment_depth_)
        points_greater_depth++;

      avg_depth = avg_depth + point[2];
    }
    avg_depth = avg_depth/segment_size;
    float frac_greater_depth = points_greater_depth/segment_size;
    float frac_lesser_depth = points_lesser_depth/segment_size;
    if(frac_greater_depth > 0.3 || frac_lesser_depth > 0.3 || avg_depth < min_segment_depth_ || avg_depth > max_segment_depth_)
    {
      ROS_INFO_STREAM("Segment depth outside valid range. frac_greater_depth: "<<frac_greater_depth<<"\t frac_lesser_depth: "<<frac_lesser_depth<<"\t avg_depth: "<<avg_depth); 
      return false;
    }
    else
    {
      return true;
    }
  }

  void publish_segments(
      const std::vector<depth_segmentation::Segment>& segments,
      const std_msgs::Header& header) {
    CHECK_GT(segments.size(), 0u);
    // Just for rviz also publish the whole scene, as otherwise only ~10
    // segments are shown:
    // https://github.com/ros-visualization/rviz/issues/689
    sensor_msgs::PointCloud2 pcl2_msg;

    if (params_.semantic_instance_segmentation.enable) {
      pcl::PointCloud<PointSurfelLabel>::Ptr scene_pcl(
          new pcl::PointCloud<PointSurfelLabel>);
      pcl::PointCloud<pcl::PointXYZL>::Ptr scene_pcl_xyzl(
          new pcl::PointCloud<pcl::PointXYZL>);
      for (depth_segmentation::Segment segment : segments) {
        if(forward_labeled_segments_only_ && segment.is_pepper == false)
          continue;
        if(is_segment_depth_valid(segment) == false)
        {
          continue;
        }
        if(segment.is_pepper)
          ROS_INFO_STREAM("Publishing instance segment: "<<int32_t(*segment.instance_label.begin())<<" with semantic label: "<<int16_t(*segment.semantic_label.begin()));
        CHECK_GT(segment.points.size(), 0u);
        pcl::PointCloud<PointSurfelLabel>::Ptr segment_pcl(
            new pcl::PointCloud<PointSurfelLabel>);
        for (std::size_t i = 0u; i < segment.points.size(); ++i) {
          PointSurfelLabel point_pcl;
          uint8_t semantic_label = 0u;
          uint32_t instance_label = 0u;
          if (segment.instance_label.size() > 0u) {
            instance_label = *(segment.instance_label.begin());
          }
          if (segment.semantic_label.size() > 0u)
          {
            semantic_label = *(segment.semantic_label.begin());
          }
          fillPoint(segment.points[i], segment.normals[i],
                    segment.original_colors[i], semantic_label, instance_label,
                    &point_pcl);

          segment_pcl->push_back(point_pcl);

          if (publish_scene_as_xyzl_)
          {
            pcl::PointXYZL point_xyzl;
            point_xyzl.x = point_pcl.x;
            point_xyzl.y = point_pcl.y;
            point_xyzl.z = point_pcl.z;
            point_xyzl.label = point_pcl.semantic_label;
            scene_pcl_xyzl->push_back(point_xyzl);
          }
          else
          {
            scene_pcl->push_back(point_pcl);
          }
        }

        sensor_msgs::PointCloud2 pcl2_msg;
        pcl::toROSMsg(*segment_pcl, pcl2_msg);
        pcl2_msg.header.stamp = header.stamp;
        pcl2_msg.header.frame_id = header.frame_id;
        point_cloud2_segment_pub_.publish(pcl2_msg);
      }
      if (params_.visualize_segmented_scene) {
        if (publish_scene_as_xyzl_)
        {
          pcl::toROSMsg(*scene_pcl_xyzl, pcl2_msg);
        }
        else
        {
          pcl::toROSMsg(*scene_pcl, pcl2_msg);
        }
      }
    } else {
      pcl::PointCloud<pcl::PointSurfel>::Ptr scene_pcl(
          new pcl::PointCloud<pcl::PointSurfel>);
      for (depth_segmentation::Segment segment : segments) {
        CHECK_GT(segment.points.size(), 0u);
        pcl::PointCloud<pcl::PointSurfel>::Ptr segment_pcl(
            new pcl::PointCloud<pcl::PointSurfel>);
        for (std::size_t i = 0u; i < segment.points.size(); ++i) {
          pcl::PointSurfel point_pcl;

          fillPoint(segment.points[i], segment.normals[i],
                    segment.original_colors[i], &point_pcl);

          segment_pcl->push_back(point_pcl);
          scene_pcl->push_back(point_pcl);
        }
        sensor_msgs::PointCloud2 pcl2_msg;
        pcl::toROSMsg(*segment_pcl, pcl2_msg);
        pcl2_msg.header.stamp = header.stamp;
        pcl2_msg.header.frame_id = header.frame_id;
        point_cloud2_segment_pub_.publish(pcl2_msg);
      }
      if (params_.visualize_segmented_scene) {
        pcl::toROSMsg(*scene_pcl, pcl2_msg);
      }
    }

    if (params_.visualize_segmented_scene) {
      pcl2_msg.header.stamp = header.stamp;
      pcl2_msg.header.frame_id = header.frame_id;
      point_cloud2_scene_pub_.publish(pcl2_msg);
    }
  }


#ifdef MASKRCNNROS_AVAILABLE
  void semanticInstanceSegmentationFromRosMsg(
      const mask_rcnn_ros::Result::ConstPtr& segmentation_msg,
      depth_segmentation::SemanticInstanceSegmentation*
          semantic_instance_segmentation) 
  {
    //ROS_WARN_STREAM("No of masks: "<<segmentation_msg->masks.size());
    semantic_instance_segmentation->masks.reserve(
        segmentation_msg->masks.size());
    semantic_instance_segmentation->labels.reserve(
        segmentation_msg->masks.size());
    for (size_t i = 0u; i < segmentation_msg->masks.size(); ++i) {
      //ROS_INFO_STREAM("\t Mask class name: "<<segmentation_msg->class_names[i]<<"\t instance id: "<<segmentation_msg->instance_ids[i]);
      /*if(segmentation_msg->class_names[i] == "peduncle" || segmentation_msg->class_names[i] == "stem")
      {
        ROS_WARN_STREAM("Peduncle mask Discarding: "<<segmentation_msg->class_names[i]<<" class_id: "<<segmentation_msg->class_ids[i]);
        continue;
      }*/
      cv_bridge::CvImagePtr cv_mask_image;
      cv_mask_image = cv_bridge::toCvCopy(segmentation_msg->masks[i],
                                          sensor_msgs::image_encodings::MONO8);
      semantic_instance_segmentation->masks.push_back(
          cv_mask_image->image.clone());
      semantic_instance_segmentation->labels.push_back(
          segmentation_msg->class_ids[i]);
      semantic_instance_segmentation->instance_ids.push_back(segmentation_msg->instance_ids[i]);
    }
  }
#endif

  void preprocess(const sensor_msgs::Image::ConstPtr& depth_msg,
                  const sensor_msgs::Image::ConstPtr& rgb_msg,
                  cv::Mat* rescaled_depth, cv::Mat* dilated_rescaled_depth,
                  cv_bridge::CvImagePtr cv_rgb_image,
                  cv_bridge::CvImagePtr cv_depth_image, cv::Mat* bw_image,
                  cv::Mat* mask) {
    CHECK_NOTNULL(rescaled_depth);
    CHECK_NOTNULL(dilated_rescaled_depth);
    CHECK(cv_rgb_image);
    CHECK(cv_depth_image);
    CHECK_NOTNULL(bw_image);
    CHECK_NOTNULL(mask);

    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      cv_depth_image = cv_bridge::toCvCopy(
          depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      
      cv::Scalar mean, stddev;
      cv::meanStdDev((cv_depth_image->image), mean, stddev);
      //ROS_WARN_STREAM_THROTTLE(1.0, "TYPE_16UC1 image " << " mean: " << mean[0] << ", stddev: " << stddev[0]);
      *rescaled_depth = cv::Mat::zeros(cv_depth_image->image.size(), CV_32FC1);
      cv::rgbd::rescaleDepth(cv_depth_image->image, CV_32FC1, *rescaled_depth);
      //ROS_WARN_STREAM_THROTTLE(1.0, "TYPE_16UC1");
    } else if (depth_msg->encoding ==
               sensor_msgs::image_encodings::TYPE_32FC1) {
      cv_depth_image = cv_bridge::toCvCopy(
          depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
      //ROS_WARN_STREAM_THROTTLE(1.0, "TYPE_32FC1");
      *rescaled_depth = cv_depth_image->image;
    } else {
      LOG(FATAL) << "Unknown depth image encoding.";
    }
    
    constexpr double kZeroValue = 0.0;
    //cv::Mat nan_mask = *rescaled_depth != *rescaled_depth;
    //rescaled_depth->setTo(kZeroValue, nan_mask);
    
    // Iterate through the matrix
    for (int r = 0; r < rescaled_depth->rows; ++r)
    {
        for (int c = 0; c < rescaled_depth->cols; ++c)
        {
            float value = (*rescaled_depth).at<float>(r, c); // Use at<double> for CV_64FC1
            if (std::isnan(value))
            {
                (*rescaled_depth).at<float>(r, c) = 0.0f; // Use 0.0 or 0.0f for CV_32FC1
            }
        }
    }
    cv::Scalar mean, stddev;
    cv::meanStdDev(*rescaled_depth, mean, stddev);
    //ROS_WARN_STREAM_THROTTLE(1.0, "rescaled_depth " << " mean: " << mean[0] << ", stddev: " << stddev[0]);


    if (params_.dilate_depth_image) {
      cv::Mat element = cv::getStructuringElement(
          cv::MORPH_RECT, cv::Size(2u * params_.dilation_size + 1u,
                                   2u * params_.dilation_size + 1u));
      cv::morphologyEx(*rescaled_depth, *dilated_rescaled_depth,
                       cv::MORPH_DILATE, element);
    } else {
      *dilated_rescaled_depth = *rescaled_depth;
    }

    *bw_image = cv::Mat::zeros(cv_rgb_image->image.size(), CV_8UC1);

    cvtColor(cv_rgb_image->image, *bw_image, cv::COLOR_RGB2GRAY);
    cv::Scalar mean1, stddev1;
    cv::meanStdDev(*bw_image, mean1, stddev1);
    //ROS_WARN_STREAM_THROTTLE(1.0, "bw_image " << " mean: " << mean1[0] << ", stddev: " << stddev1[0]);

    *mask = cv::Mat::zeros(bw_image->size(), CV_8UC1);
    mask->setTo(cv::Scalar(depth_segmentation::CameraTracker::kImageRange));
  }

  void computeEdgeMap(const sensor_msgs::Image::ConstPtr& depth_msg,
                      const sensor_msgs::Image::ConstPtr& rgb_msg,
                      cv::Mat& rescaled_depth,
                      cv_bridge::CvImagePtr cv_rgb_image,
                      cv_bridge::CvImagePtr cv_depth_image, cv::Mat& bw_image,
                      cv::Mat& mask, cv::Mat* depth_map, cv::Mat* normal_map,
                      cv::Mat* edge_map) {
#ifdef WRITE_IMAGES
    cv::imwrite(
        std::to_string(cv_rgb_image->header.stamp.toSec()) + "_rgb_image.png",
        cv_rgb_image->image);
    cv::imwrite(
        std::to_string(cv_rgb_image->header.stamp.toSec()) + "_bw_image.png",
        bw_image);
    cv::imwrite(
        std::to_string(depth_msg->header.stamp.toSec()) + "_depth_image.png",
        rescaled_depth);
    cv::imwrite(
        std::to_string(depth_msg->header.stamp.toSec()) + "_depth_mask.png",
        mask);
#endif  // WRITE_IMAGES

#ifdef DISPLAY_DEPTH_IMAGES
    camera_tracker_.visualize(camera_tracker_.getDepthImage(), rescaled_depth);
#endif  // DISPLAY_DEPTH_IMAGES

    // Compute transform from tracker.
    if (depth_segmentation::kUseTracker) {
      if (camera_tracker_.computeTransform(bw_image, rescaled_depth, mask)) {
        publish_tf(camera_tracker_.getWorldTransform(),
                   depth_msg->header.stamp);
      } else {
        LOG(ERROR) << "Failed to compute Transform.";
      }
    }

    *depth_map = cv::Mat::zeros(depth_camera_.getWidth(),
                                depth_camera_.getHeight(), CV_32FC3);
    depth_segmenter_.computeDepthMap(rescaled_depth, depth_map);

    // Compute normal map.
    *normal_map = cv::Mat::zeros(depth_map->size(), CV_32FC3);

    if (params_.normals.method ==
            depth_segmentation::SurfaceNormalEstimationMethod::kFals ||
        params_.normals.method ==
            depth_segmentation::SurfaceNormalEstimationMethod::kSri ||
        params_.normals.method ==
            depth_segmentation::SurfaceNormalEstimationMethod::
                kDepthWindowFilter) {
      depth_segmenter_.computeNormalMap(*depth_map, normal_map);
    } else if (params_.normals.method ==
               depth_segmentation::SurfaceNormalEstimationMethod::kLinemod) {
      depth_segmenter_.computeNormalMap(cv_depth_image->image, normal_map);
    }

    // Compute depth discontinuity map.
    cv::Mat discontinuity_map = cv::Mat::zeros(
        depth_camera_.getWidth(), depth_camera_.getHeight(), CV_32FC1);
    if (params_.depth_discontinuity.use_discontinuity) {
      depth_segmenter_.computeDepthDiscontinuityMap(rescaled_depth,
                                                    &discontinuity_map);
    }

    // Compute maximum distance map.
    cv::Mat distance_map = cv::Mat::zeros(depth_camera_.getWidth(),
                                          depth_camera_.getHeight(), CV_32FC1);
    if (params_.max_distance.use_max_distance) {
      depth_segmenter_.computeMaxDistanceMap(*depth_map, &distance_map);
    }

    // Compute minimum convexity map.
    cv::Mat convexity_map = cv::Mat::zeros(depth_camera_.getWidth(),
                                           depth_camera_.getHeight(), CV_32FC1);
    if (params_.min_convexity.use_min_convexity) {
      depth_segmenter_.computeMinConvexityMap(*depth_map, *normal_map,
                                              &convexity_map);
    }

    // Compute final edge map.
    *edge_map = cv::Mat::zeros(depth_camera_.getWidth(),
                               depth_camera_.getHeight(), CV_32FC1);
    depth_segmenter_.computeFinalEdgeMap(convexity_map, distance_map,
                                         discontinuity_map, edge_map);
  }

  void imageCallback(const sensor_msgs::Image::ConstPtr& depth_msg,
                     const sensor_msgs::Image::ConstPtr& rgb_msg) {
    if (camera_info_ready_) {
      cv_bridge::CvImagePtr cv_rgb_image(new cv_bridge::CvImage);
      cv_rgb_image = cv_bridge::toCvCopy(rgb_msg, rgb_msg->encoding);
      if (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) {
        cv::cvtColor(cv_rgb_image->image, cv_rgb_image->image, CV_BGR2RGB);
      }

      cv_bridge::CvImagePtr cv_depth_image(new cv_bridge::CvImage);
      cv::Mat rescaled_depth, dilated_rescaled_depth, bw_image, mask, depth_map,
          normal_map, edge_map;
      preprocess(depth_msg, rgb_msg, &rescaled_depth, &dilated_rescaled_depth,
                 cv_rgb_image, cv_depth_image, &bw_image, &mask);
      if (!camera_tracker_.getRgbImage().empty() &&
              !camera_tracker_.getDepthImage().empty() ||
          !depth_segmentation::kUseTracker) {
        computeEdgeMap(depth_msg, rgb_msg, dilated_rescaled_depth, cv_rgb_image,
                       cv_depth_image, bw_image, mask, &depth_map, &normal_map,
                       &edge_map);

        cv::Mat label_map(edge_map.size(), CV_32FC1);
        cv::Mat remove_no_values =
            cv::Mat::zeros(edge_map.size(), edge_map.type());
        edge_map.copyTo(remove_no_values,
                        dilated_rescaled_depth == dilated_rescaled_depth);
        edge_map = remove_no_values;
        std::vector<depth_segmentation::Segment> segments;
        std::vector<cv::Mat> segment_masks;

        depth_segmenter_.labelMap(cv_rgb_image->image, rescaled_depth,
                                  depth_map, edge_map, normal_map, &label_map,
                                  &segment_masks, &segments);

        if (segments.size() > 0u) {
          publish_segments(segments, depth_msg->header);
        }
      }
      // Update the member images to the new images.
      // TODO(ff): Consider only doing this, when we are far enough away
      // from a frame. (Which basically means we would set a keyframe.)
      depth_camera_.setImage(rescaled_depth);
      depth_camera_.setMask(mask);
      rgb_camera_.setImage(bw_image);
    }
  }

#ifdef MASKRCNNROS_AVAILABLE
  void imageSegmentationCallback(
      const sensor_msgs::Image::ConstPtr& depth_msg,
      const sensor_msgs::Image::ConstPtr& rgb_msg,
      const mask_rcnn_ros::Result::ConstPtr& segmentation_msg) //,
      //const sensor_msgs::PointCloud2::ConstPtr& pc2_msg) 
  {
    depth_segmentation::SemanticInstanceSegmentation instance_segmentation;
    ROS_WARN_STREAM_THROTTLE(0.5, "imageSegmentationCallback");
    semanticInstanceSegmentationFromRosMsg(segmentation_msg,
                                           &instance_segmentation);
    
    sensor_msgs::PointCloud2::ConstPtr pc2_msg = NULL;
    // if(pc2_msg == NULL)
    //   LOG(WARNING)<<"Point Cloud2 msg NULL";
    // else
    //   ROS_WARN_STREAM_THROTTLE(10, "point cloud frame id: "<<pc2_msg->header.frame_id);
    int best_depth_img_idx = -1;
    stable_depth_image_available_ = false;
    robot_moving_ = true;
    robot_steady_ = false;
    if(publish_while_moving_ == false)
    {
      ROS_WARN_STREAM_THROTTLE(10.0, "Publish while steady");
      robot_moving_ = isRobotMoving(joint_state_, prev_joint_state_, depth_msg, use_joint_velocities_, use_selective_joints_, selective_joint_names_, max_joint_velocity_, max_joint_difference_, world_frame_, use_transform_, last_robot_moved_time_);
      robot_steady_ = isRobotSteady(robot_moving_, last_robot_moved_time_, wait_time_stationary_, last_robot_steady_time_);
    }
    else
    {
      ROS_WARN_STREAM_THROTTLE(10.0, "Publish while moving");
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud((new pcl::PointCloud<pcl::PointXYZRGB>));
    pcl_cloud = NULL;
    cv_bridge::CvImagePtr cv_rgb_image(new cv_bridge::CvImage);
    cv_bridge::CvImagePtr cv_depth_image(new cv_bridge::CvImage);
    cv::Mat rescaled_depth, dilated_rescaled_depth, bw_image, mask, depth_map, normal_map, edge_map;

    if (robot_steady_ || publish_while_moving_) 
    {
      if (camera_info_ready_) {
        //ROS_WARN_STREAM_THROTTLE(0.1, "preprocess: "<<robot_steady_<<" last_robot_moved_time: "<<last_robot_moved_time_);
        //cv_bridge::CvImagePtr cv_rgb_image(new cv_bridge::CvImage);
        cv_rgb_image = cv_bridge::toCvCopy(rgb_msg, rgb_msg->encoding);
        if (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) {
          cv::cvtColor(cv_rgb_image->image, cv_rgb_image->image, CV_BGR2RGB);
        }
        preprocess(depth_msg, rgb_msg, &rescaled_depth, &dilated_rescaled_depth,
                  cv_rgb_image, cv_depth_image, &bw_image, &mask);
        
        if(publish_while_moving_ == false)
        {
          if(use_stability_score_ && robot_steady_ && depth_segmenter_.getVectorSize() < 5)
          {
            //ROS_WARN_STREAM_THROTTLE(0.2, "Adding to vectors");
            depth_segmenter_.addToVectors( &rescaled_depth, &dilated_rescaled_depth,
                  cv_rgb_image, cv_depth_image, &bw_image, &mask);
            stable_depth_image_available_ = false;
          }
          else
          {
            //ROS_WARN_STREAM_THROTTLE(0.2, "image acquired");
            stable_depth_image_available_ = robot_steady_;
          }
        }
      }
    }
    if(use_stability_score_ && ((depth_segmenter_.getVectorSize() > 1 && robot_moving_) || (depth_segmenter_.getVectorSize() > 4)))
    {
      //ROS_WARN_STREAM_THROTTLE(1.0, "Loop entered");
      {
        //ROS_WARN_STREAM_THROTTLE(0.5, "Selecting best depth image");
        best_depth_img_idx = depth_segmenter_.selectBestDepthImage();
        //ROS_WARN_STREAM_THROTTLE(0.5, "Returned from select best depth image");
        if(best_depth_img_idx < 0)
        {
          ROS_ERROR_STREAM_THROTTLE(1.0, "No stable image");
          stable_depth_image_available_ = false;
          return;
        }
        //ROS_WARN_STREAM_THROTTLE(0.5, "Selected best depth image"<<best_depth_img_idx);
        stable_depth_image_available_ = depth_segmenter_.retrieveFromVectors(best_depth_img_idx, &rescaled_depth, &dilated_rescaled_depth, cv_rgb_image, cv_depth_image, &bw_image, &mask);
      }
    }
  
    if (camera_info_ready_ && ((publish_while_moving_ == false && (stable_depth_image_available_ == true ) ) || (publish_while_moving_ == true)))
    {
      //ROS_WARN_THROTTLE(1.0, "Retrieved image for further processing");
      if (!camera_tracker_.getRgbImage().empty() &&
            !camera_tracker_.getDepthImage().empty() ||
        !depth_segmentation::kUseTracker) {
        computeEdgeMap(depth_msg, rgb_msg, dilated_rescaled_depth, cv_rgb_image,
                      cv_depth_image, bw_image, mask, &depth_map, &normal_map,
                      &edge_map);

        cv::Mat label_map(edge_map.size(), CV_32FC1);
        cv::Mat remove_no_values =
            cv::Mat::zeros(edge_map.size(), edge_map.type());
        edge_map.copyTo(remove_no_values,
                        dilated_rescaled_depth == dilated_rescaled_depth);
        edge_map = remove_no_values;
        std::vector<depth_segmentation::Segment> segments;
        std::vector<depth_segmentation::Segment> overlap_segments;
        std::vector<cv::Mat> segment_masks;

        //ROS_INFO_STREAM_THROTTLE(5.0, "Before if publishing segments"); 
        if(use_overlap_bits_only_)
        {
          depth_segmenter_.labelMap(cv_rgb_image->image, rescaled_depth,
                                instance_segmentation, depth_map, edge_map,
                                normal_map, &label_map, &segment_masks,
                                &segments, overlap_segments, pcl_cloud);
          if (overlap_segments.size() > 0u) {
              ROS_INFO_STREAM_THROTTLE(10.0, "Before publishing segments"); 
              publish_segments(overlap_segments, depth_msg->header);
          }
        }
        else
        {
          depth_segmenter_.labelMap(cv_rgb_image->image, rescaled_depth,
                                  instance_segmentation, depth_map, edge_map,
                                  normal_map, &label_map, &segment_masks,
                                  &segments, pcl_cloud);
          if (segments.size() > 0u) {
            ROS_WARN_STREAM_THROTTLE(10.0, "Before publishing segments"); 
            publish_segments(segments, depth_msg->header);
          }
        }
        depth_camera_.setImage(rescaled_depth);
        depth_camera_.setMask(mask);
        rgb_camera_.setImage(bw_image);
      }
    }
    return;
    // Update the member images to the new images.
    // TODO(ff): Consider only doing this, when we are far enough away
    // from a frame. (Which basically means we would set a keyframe.)
    
  }
#endif

  void cameraInfoCallback(
      const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info_msg,
      const sensor_msgs::CameraInfo::ConstPtr& rgb_camera_info_msg) {
    if (camera_info_ready_) {
      return;
    }

    sensor_msgs::CameraInfo depth_info;
    depth_info = *depth_camera_info_msg;
    Eigen::Vector2d depth_image_size(depth_info.width, depth_info.height);

    cv::Mat K_depth = cv::Mat::eye(3, 3, CV_32FC1);
    K_depth.at<float>(0, 0) = depth_info.K[0];
    K_depth.at<float>(0, 2) = depth_info.K[2];
    K_depth.at<float>(1, 1) = depth_info.K[4];
    K_depth.at<float>(1, 2) = depth_info.K[5];
    K_depth.at<float>(2, 2) = depth_info.K[8];

    depth_camera_.initialize(depth_image_size.x(), depth_image_size.y(),
                             CV_32FC1, K_depth);

    sensor_msgs::CameraInfo rgb_info;
    rgb_info = *rgb_camera_info_msg;
    Eigen::Vector2d rgb_image_size(rgb_info.width, rgb_info.height);

    cv::Mat K_rgb = cv::Mat::eye(3, 3, CV_32FC1);
    K_rgb.at<float>(0, 0) = rgb_info.K[0];
    K_rgb.at<float>(0, 2) = rgb_info.K[2];
    K_rgb.at<float>(1, 1) = rgb_info.K[4];
    K_rgb.at<float>(1, 2) = rgb_info.K[5];
    K_rgb.at<float>(2, 2) = rgb_info.K[8];

    rgb_camera_.initialize(rgb_image_size.x(), rgb_image_size.y(), CV_8UC1,
                           K_rgb);

    depth_segmenter_.initialize();
    camera_tracker_.initialize(
        camera_tracker_.kCameraTrackerNames
            [camera_tracker_.CameraTrackerType::kRgbdICPOdometry]);

    camera_info_ready_ = true;
  }

  bool checkTfMoved(const sensor_msgs::ImageConstPtr& depth_msg, const std::string& world_frame, bool use_transform, ros::Time& tf_moved_time) 
  {
    if (!use_transform) {
      return false;
    }
    
    static Eigen::Isometry3d last_tf_eigen_;

    geometry_msgs::TransformStamped pc_frame_tf;
    try
    {
      pc_frame_tf = tf_buffer_->lookupTransform(world_frame, depth_msg->header.frame_id, depth_msg->header.stamp);
      ROS_INFO_STREAM_THROTTLE(20, "Transform from " << depth_msg->header.frame_id << " to " << world_frame << " for time " << depth_msg->header.stamp << " found");
    }
    catch (const tf2::TransformException& e)
    {
      ROS_ERROR_STREAM("Couldn't find transform to map frame: " << e.what());
      return false;
    }

    Eigen::Isometry3d tf_eigen = tf2::transformToEigen(pc_frame_tf);
    if (!tf_eigen.isApprox(last_tf_eigen_, 1e-2))
    {
      tf_moved_time = ros::Time::now();
      ROS_WARN_THROTTLE(1, "tf_eigen and last_tf_eigen are not approx equal");
      last_tf_eigen_ = tf_eigen;
      return true;
    }

    last_tf_eigen_ = tf_eigen;
    return false;
  }

  bool checkJointVelocities(const sensor_msgs::JointState& joint_state, double max_joint_velocity, 
                            bool use_selective_joints, const std::vector<std::string>& selective_joint_names,                          
                            ros::Time& last_vel_exceeded_time) 
  {
    bool vel_exceeded = false;
    for (size_t i = 0; i < joint_state.velocity.size(); ++i)
    {
      // Selective joint check
      if (use_selective_joints)
      {
        if (std::find(selective_joint_names.begin(), selective_joint_names.end(), joint_state.name[i]) == selective_joint_names.end())
        {
          continue; // Skip this joint if it's not in the selective joints list
        }
      }

      // Check if velocity exceeds the threshold
      if (joint_state.velocity[i] > max_joint_velocity)
      {
        ROS_WARN_STREAM_THROTTLE(1, "Joint velocity of at least one joint is greater than threshold: " 
          << joint_state.velocity[i] << "\t max allowed: " << max_joint_velocity);
        last_vel_exceeded_time = joint_state.header.stamp;
        vel_exceeded = true;
      }
    }
    return vel_exceeded;
  }

  bool checkJointPositionDifference(const sensor_msgs::JointState& joint_state,const sensor_msgs::JointState& prev_joint_state,
                                    double max_joint_difference, ros::Time& last_pos_exceeded_time) 
  {
    double sq_sum = 0;
    for (size_t i = 0; i < joint_state.position.size(); ++i)
    {
      sq_sum += std::pow(joint_state.position[i] - prev_joint_state.position[i], 2);
    }
    double dist_norm = std::sqrt(sq_sum);

    if (dist_norm > max_joint_difference)
    {
      last_pos_exceeded_time = joint_state.header.stamp;
      ROS_WARN_STREAM_THROTTLE(1, "Joint distance norm is greater than threshold: " << dist_norm);
      return true;
    }

    //ROS_WARN_STREAM_THROTTLE(20, "Joint distance norm: " << dist_norm);
    return false;
  }

  bool isRobotMoving(const sensor_msgs::JointState& joint_state, const sensor_msgs::JointState& prev_joint_state, const sensor_msgs::ImageConstPtr& depth_msg,
                   bool use_joint_velocities, bool use_selective_joints, const std::vector<std::string>& selective_joint_names, double max_joint_velocity, 
                   double max_joint_difference, const std::string& world_frame, bool use_transform, ros::Time& last_robot_moved_time) 
  {
    bool robot_moving = false;

    // Initialize times
    ros::Time tf_moved_time, last_vel_exceeded_time, last_pos_diff_exceeded_time;
    bool tf_moved = false;
    bool vel_exceeded = false;
    bool pos_diff_exceeded = false;

    // Check if transformation has moved
    if (use_transform) {
      tf_moved = checkTfMoved(depth_msg, world_frame, use_transform, tf_moved_time);
    }

    
    if (use_joint_velocities) 
    {
      // Check if joint velocities exceed the threshold
      vel_exceeded = checkJointVelocities(joint_state, max_joint_velocity, use_selective_joints, selective_joint_names, last_vel_exceeded_time);
    }
    else
    {
    // Check if the joint position difference exceeds the threshold
      checkJointPositionDifference(joint_state, prev_joint_state, max_joint_difference, last_pos_diff_exceeded_time);
    }

    // Determine the latest time and update robot moving status
    if (tf_moved && vel_exceeded) {
      // Both tf_moved and vel_exceeded are true, use the latest timestamp
      last_robot_moved_time = std::max(tf_moved_time, last_vel_exceeded_time);
      robot_moving = true;
    } else if (tf_moved) {
      last_robot_moved_time = tf_moved_time;
      robot_moving = true;
    } else if (vel_exceeded) {
      last_robot_moved_time = last_vel_exceeded_time;
      robot_moving = true;
    } else if (pos_diff_exceeded) {
      robot_moving = true;
    }
    ROS_WARN_STREAM_THROTTLE(10, "Robot moving status: "<<robot_moving);
    return robot_moving;
  }

bool isRobotSteady( bool robot_moving, const ros::Time& last_robot_moved_time, double wait_time_stationary, ros::Time& last_robot_steady_time) 
{
  if (!robot_moving && fabs((ros::Time::now() - last_robot_moved_time).toSec()) > wait_time_stationary) 
  {
    ROS_WARN_STREAM_THROTTLE(10, "Robot steady");
    last_robot_steady_time = ros::Time::now();
    return true;
  } 
  else 
  {
    ROS_WARN_STREAM_THROTTLE(1.0, "Robot still not steady hence not processing");
    return false;
  }
}

  
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;

  LOG(INFO) << "Starting depth segmentation ... ";
  ros::init(argc, argv, "depth_segmentation_node");
  DepthSegmentationNode depth_segmentation_node;

  dynamic_reconfigure::Server<depth_segmentation::DepthSegmenterConfig>
      reconfigure_server;
  dynamic_reconfigure::Server<depth_segmentation::DepthSegmenterConfig>::
      CallbackType dynamic_reconfigure_function;

  dynamic_reconfigure_function = boost::bind(
      &depth_segmentation::DepthSegmenter::dynamicReconfigureCallback,
      &depth_segmentation_node.depth_segmenter_, _1, _2);
  reconfigure_server.setCallback(dynamic_reconfigure_function);

  while (ros::ok()) {
    ros::spin();
  }

  return EXIT_SUCCESS;
}

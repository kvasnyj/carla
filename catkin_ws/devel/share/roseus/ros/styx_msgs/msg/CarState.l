;; Auto-generated. Do not edit!


(when (boundp 'styx_msgs::CarState)
  (if (not (find-package "STYX_MSGS"))
    (make-package "STYX_MSGS"))
  (shadow 'CarState (find-package "STYX_MSGS")))
(unless (find-package "STYX_MSGS::CARSTATE")
  (make-package "STYX_MSGS::CARSTATE"))

(in-package "ROS")
;;//! \htmlinclude CarState.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass styx_msgs::CarState
  :super ros::object
  :slots (_header _position _speed _camera1d ))

(defmethod styx_msgs::CarState
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:position __position) 0.0)
    ((:speed __speed) 0.0)
    ((:camera1d __camera1d) (make-array 0 :initial-element 0 :element-type :char))
    )
   (send-super :init)
   (setq _header __header)
   (setq _position (float __position))
   (setq _speed (float __speed))
   (setq _camera1d __camera1d)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:position
   (&optional __position)
   (if __position (setq _position __position)) _position)
  (:speed
   (&optional __speed)
   (if __speed (setq _speed __speed)) _speed)
  (:camera1d
   (&optional __camera1d)
   (if __camera1d (setq _camera1d __camera1d)) _camera1d)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; float32 _position
    4
    ;; float32 _speed
    4
    ;; uint8[] _camera1d
    (* 1    (length _camera1d)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; float32 _position
       (sys::poke _position (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _speed
       (sys::poke _speed (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; uint8[] _camera1d
     (write-long (length _camera1d) s)
     (princ _camera1d s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; float32 _position
     (setq _position (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _speed
     (setq _speed (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; uint8[] _camera1d
   (let ((n (sys::peek buf ptr- :integer))) (incf ptr- 4)
     (setq _camera1d (make-array n :element-type :char))
     (replace _camera1d buf :start2 ptr-) (incf ptr- n))
   ;;
   self)
  )

(setf (get styx_msgs::CarState :md5sum-) "e925c87f59560d5f1ad5982fd69710ba")
(setf (get styx_msgs::CarState :datatype-) "styx_msgs/CarState")
(setf (get styx_msgs::CarState :definition-)
      "Header header
float32 position
float32 speed
uint8[] camera1d
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

")



(provide :styx_msgs/CarState "e925c87f59560d5f1ad5982fd69710ba")



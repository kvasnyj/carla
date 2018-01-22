; Auto-generated. Do not edit!


(cl:in-package styx_msgs-msg)


;//! \htmlinclude CarState.msg.html

(cl:defclass <CarState> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (position
    :reader position
    :initarg :position
    :type cl:float
    :initform 0.0)
   (speed
    :reader speed
    :initarg :speed
    :type cl:float
    :initform 0.0)
   (camera1d
    :reader camera1d
    :initarg :camera1d
    :type (cl:vector cl:integer)
   :initform (cl:make-array 0 :element-type 'cl:integer :initial-element 0)))
)

(cl:defclass CarState (<CarState>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CarState>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CarState)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name styx_msgs-msg:<CarState> is deprecated: use styx_msgs-msg:CarState instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <CarState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader styx_msgs-msg:header-val is deprecated.  Use styx_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'position-val :lambda-list '(m))
(cl:defmethod position-val ((m <CarState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader styx_msgs-msg:position-val is deprecated.  Use styx_msgs-msg:position instead.")
  (position m))

(cl:ensure-generic-function 'speed-val :lambda-list '(m))
(cl:defmethod speed-val ((m <CarState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader styx_msgs-msg:speed-val is deprecated.  Use styx_msgs-msg:speed instead.")
  (speed m))

(cl:ensure-generic-function 'camera1d-val :lambda-list '(m))
(cl:defmethod camera1d-val ((m <CarState>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader styx_msgs-msg:camera1d-val is deprecated.  Use styx_msgs-msg:camera1d instead.")
  (camera1d m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CarState>) ostream)
  "Serializes a message object of type '<CarState>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'speed))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'camera1d))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:write-byte (cl:ldb (cl:byte 8 0) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 8) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 16) ele) ostream)
  (cl:write-byte (cl:ldb (cl:byte 8 24) ele) ostream))
   (cl:slot-value msg 'camera1d))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CarState>) istream)
  "Deserializes a message object of type '<CarState>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'position) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'speed) (roslisp-utils:decode-single-float-bits bits)))
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'camera1d) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'camera1d)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) (cl:aref vals i)) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) (cl:aref vals i)) (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CarState>)))
  "Returns string type for a message object of type '<CarState>"
  "styx_msgs/CarState")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CarState)))
  "Returns string type for a message object of type 'CarState"
  "styx_msgs/CarState")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CarState>)))
  "Returns md5sum for a message object of type '<CarState>"
  "5f5ce100f2337443c1c11d6317e57dfc")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CarState)))
  "Returns md5sum for a message object of type 'CarState"
  "5f5ce100f2337443c1c11d6317e57dfc")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CarState>)))
  "Returns full string definition for message of type '<CarState>"
  (cl:format cl:nil "Header header~%float32 position~%float32 speed~%uint32[] camera1d~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CarState)))
  "Returns full string definition for message of type 'CarState"
  (cl:format cl:nil "Header header~%float32 position~%float32 speed~%uint32[] camera1d~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%# 0: no frame~%# 1: global frame~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CarState>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4
     4
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'camera1d) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CarState>))
  "Converts a ROS message object to a list"
  (cl:list 'CarState
    (cl:cons ':header (header msg))
    (cl:cons ':position (position msg))
    (cl:cons ':speed (speed msg))
    (cl:cons ':camera1d (camera1d msg))
))

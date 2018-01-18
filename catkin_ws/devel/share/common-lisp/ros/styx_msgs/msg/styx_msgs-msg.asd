
(cl:in-package :asdf)

(defsystem "styx_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "CarState" :depends-on ("_package_CarState"))
    (:file "_package_CarState" :depends-on ("_package"))
  ))
import rospy
# from human_pose.msg import Human_Pose

class RosHandler:

    def initROSPublisher(self):

        # ROS Publisher init
        self.rosPub = rospy.Publisher("human_pose_id0", Human_Pose, queue_size = 54)
        # ROS node init
        rospy.init_node('human_pose_pub_node', anonymous=True)
        # Frequency init
        self.rate = rospy.Rate(30)

    def setRosMessage(self, human_pose):
        
        human_pose.kpt0.x = self.keypoints3D[id][0][0]
        human_pose.kpt0.y = self.keypoints3D[id][0][1]
        human_pose.kpt0.z = self.keypoints3D[id][0][2]

        human_pose.kpt1.x = self.keypoints3D[id][1][0]
        human_pose.kpt1.y = self.keypoints3D[id][1][1]
        human_pose.kpt1.z = self.keypoints3D[id][1][2]
        
        human_pose.kpt2.x = self.keypoints3D[id][2][0]
        human_pose.kpt2.y = self.keypoints3D[id][2][1]
        human_pose.kpt2.z = self.keypoints3D[id][2][2]
        
        human_pose.kpt3.x = self.keypoints3D[id][3][0]
        human_pose.kpt3.y = self.keypoints3D[id][3][1]
        human_pose.kpt3.z = self.keypoints3D[id][3][2]
        
        human_pose.kpt4.x = self.keypoints3D[id][4][0]
        human_pose.kpt4.y = self.keypoints3D[id][4][1]
        human_pose.kpt4.z = self.keypoints3D[id][4][2]
        
        human_pose.kpt5.x = self.keypoints3D[id][5][0]
        human_pose.kpt5.y = self.keypoints3D[id][5][1]
        human_pose.kpt5.z = self.keypoints3D[id][5][2]
        
        human_pose.kpt6.x = self.keypoints3D[id][6][0]
        human_pose.kpt6.y = self.keypoints3D[id][6][1]
        human_pose.kpt6.z = self.keypoints3D[id][6][2]
        
        human_pose.kpt7.x = self.keypoints3D[id][7][0]
        human_pose.kpt7.y = self.keypoints3D[id][7][1]
        human_pose.kpt7.z = self.keypoints3D[id][7][2]
        
        human_pose.kpt8.x = self.keypoints3D[id][8][0]
        human_pose.kpt8.y = self.keypoints3D[id][8][1]
        human_pose.kpt8.z = self.keypoints3D[id][8][2]
        
        human_pose.kpt9.x = self.keypoints3D[id][9][0]
        human_pose.kpt9.y = self.keypoints3D[id][9][1]
        human_pose.kpt9.z = self.keypoints3D[id][9][2]
        
        human_pose.kpt10.x = self.keypoints3D[id][10][0]
        human_pose.kpt10.y = self.keypoints3D[id][10][1]
        human_pose.kpt10.z = self.keypoints3D[id][10][2]
        
        human_pose.kpt11.x = self.keypoints3D[id][11][0]
        human_pose.kpt11.y = self.keypoints3D[id][11][1]
        human_pose.kpt11.z = self.keypoints3D[id][11][2]
        
        human_pose.kpt12.x = self.keypoints3D[id][12][0]
        human_pose.kpt12.y = self.keypoints3D[id][12][1]
        human_pose.kpt12.z = self.keypoints3D[id][12][2]

        human_pose.kpt13.x = self.keypoints3D[id][13][0]
        human_pose.kpt13.y = self.keypoints3D[id][13][1]
        human_pose.kpt13.z = self.keypoints3D[id][13][2]

        human_pose.kpt14.x = self.keypoints3D[id][14][0]
        human_pose.kpt14.y = self.keypoints3D[id][14][1]
        human_pose.kpt14.z = self.keypoints3D[id][14][2]

        human_pose.kpt15.x = self.keypoints3D[id][15][0]
        human_pose.kpt15.y = self.keypoints3D[id][15][1]
        human_pose.kpt15.z = self.keypoints3D[id][15][2]

        human_pose.kpt16.x = self.keypoints3D[id][16][0]
        human_pose.kpt16.y = self.keypoints3D[id][16][1]
        human_pose.kpt16.z = self.keypoints3D[id][16][2]

        human_pose.kpt17.x = self.keypoints3D[id][17][0]
        human_pose.kpt17.y = self.keypoints3D[id][17][1]
        human_pose.kpt17.z = self.keypoints3D[id][17][2]

        return human_pose

    def publishOnROS(self, id = 0, TOPICLOGFLAG = False):
        
        if rospy.is_shutdown():
            print("Waiting for roscore")
            #TODO: flag de erro

        human_pose = Human_Pose()
        
        human_pose = self.setMessage(human_pose)
        
        if(TOPICLOGFLAG):
            #rospy.loginfo("I publish:")
            rospy.loginfo(human_pose)
            
        self.rosPub.publish(human_pose)
        self.rate.sleep()
        

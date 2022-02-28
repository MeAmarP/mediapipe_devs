import cv2
import mediapipe as mp
import sys


# TODO Add exception handling, when and where it is possible
# TODO Add doc strings


class HandGestureDetector:
  """
  Detect faces in the given image/video frame and extract bonding box information.
  """

  def __init__(self,maxHands=1, minConfScore=0.5) -> None:
    """
    Initialize mediapipe hands solution used to extract hand/fingers keypoints

    Args:
        maxHands (int, optional): Max number of hands to detect in frame. Defaults to 1.
        minConfScore (float, optional): Confidence value to assert hand detection as success. Defaults to 0.5.
    """
    self.ASSERT_THRESH = 5
    self.gesture_assert_counter = 0
    self.up_count_done_flag = False
    self.down_count_done_flag = False
    self.fin_flag = False

    self.CountMaintainer = {'THUMB_UP': 0, 'THUMB_DOWN':0}

    mpHands = mp.solutions.hands
    self.HandDets = mpHands.Hands(static_image_mode=True,
                                  max_num_hands=maxHands,
                                  min_detection_confidence=minConfScore)

  def checkThumbsUpDown(self, img_frame):
    """
    Method to pre-process input image, extract keypoints of the finger tips and logic to assert hand gesture as
    THUMBS-UP or THUMBS-DOWN. Update count for current frame.

    Args:
        img_frame (np.ndarray): Input image/video frame
    """
    
    self.IMG_HT,self.IMG_WT = img_frame.shape[:2]
    img_frame = cv2.flip(img_frame, 1)
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)

    self.results = self.HandDets.process(img_frame)

    if self.results.multi_hand_landmarks:
      for h_lmarks in self.results.multi_hand_landmarks:
        thumb_tip_val = h_lmarks.landmark[4]
        index_tip_val = h_lmarks.landmark[8]
        middle_tip_val = h_lmarks.landmark[12]
        ring_tip_val = h_lmarks.landmark[16]
        pinky_tip_val = h_lmarks.landmark[20]

        if thumb_tip_val.y < index_tip_val.y < middle_tip_val.y < ring_tip_val.y < pinky_tip_val.y:          
          self.gesture_assert_counter += 1
          if self.gesture_assert_counter >= self.ASSERT_THRESH and (self.up_count_done_flag == False):                              
            self.up_count_done_flag = True

        if thumb_tip_val.y > index_tip_val.y > middle_tip_val.y > ring_tip_val.y > pinky_tip_val.y:
          self.gesture_assert_counter += 1
          if self.gesture_assert_counter >= self.ASSERT_THRESH and (self.down_count_done_flag == False):
            self.down_count_done_flag = True
    else:
      self.up_count_done_flag = False
      self.down_count_done_flag = False
      self.gesture_assert_counter = 0
      self.fin_flag = False

  def getThumbCount(self, img):
    self.checkThumbsUpDown(img)
    if self.up_count_done_flag and (self.fin_flag == False):
      self.CountMaintainer['THUMB_UP'] =  self.CountMaintainer['THUMB_UP'] + 1
      self.fin_flag = True
    if self.down_count_done_flag and (self.fin_flag == False):
      self.CountMaintainer['THUMB_DOWN'] = self.CountMaintainer['THUMB_DOWN'] + 1
      self.fin_flag = True

    return self.CountMaintainer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='To Run App >> python count_face_hand.py -v <path-to-video-file> ')
    parser.add_argument('-v', '--videosrc', default=0 ,help="Pass the path to input video file, by default reads webcam source of the PC")
    args = parser.parse_args()

    #Test class on Video
    vidcap = cv2.VideoCapture(args.videosrc)
   
    handgestvid = HandGestureDetector(maxHands=1)


    if not vidcap.isOpened():
      print("Unble to Read the Video")  
      vidcap.release()

    while vidcap.isOpened():
      ok, frame = vidcap.read()
      if not ok:
        print("===" * 20)
        print("End of the Video")
        break    
#      frame = cv2.resize(frame,(416,416))
      thumb_count = handgestvid.getThumbCount(frame)


      cv2.putText(frame, "Likes = " +str(thumb_count['THUMB_UP']), (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(128,255,0), 2)
      cv2.putText(frame, "Dis-Likes = " +str(thumb_count['THUMB_DOWN']), (10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(128,255,0), 2)
      cv2.imshow('Output', frame)
      
      if cv2.waitKey(1) == ord('q'):
          break

    vidcap.release()
    cv2.destroyAllWindows()

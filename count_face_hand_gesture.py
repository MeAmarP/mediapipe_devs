import cv2
import mediapipe as mp
import sys


# TODO Add exception handling, when and where it is possible
# TODO Add doc strings


class FaceDetector:
  """ 
  Detect faces in the given image/video frame and extract bonding box information.
  """
  def __init__(self, minConfScore=0.45) -> None:
    """ 
    Initialize mediapipe face detect solution with confidence score, by default full-range model is selected.
    
    Args:
        minConfScore (float, optional): Confidence value from face detect model, to consider as success . Defaults to 0.45.
    """
    
    self.face_count = 0    
    mpFaceDetect = mp.solutions.face_detection
    self.FaceDetect = mpFaceDetect.FaceDetection(model_selection=1,
                                            min_detection_confidence=minConfScore)
  def getDetectedFaces(self, img_frame):
    """
    Method to pre-process input frame, extract scaled bounding box values and update the face count for current frame.

    Args:
        img_frame (np.ndarray): Input image/video frame

    Returns:
        face_bboxes (list): bounding box values in [top, left, bottom, right] format
    """
    
    face_bboxes = []
    self.IMG_HT,self.IMG_WT = img_frame.shape[:2]
    img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)

    self.results = self.FaceDetect.process(img_frame)

    if self.results.detections:
      faces = [ele.location_data.relative_bounding_box for ele in self.results.detections]
      for ele_bbox in faces:
        x1 = int(ele_bbox.xmin  * self.IMG_WT)
        y1 = int(ele_bbox.ymin  * self.IMG_HT)
        x2 = int((ele_bbox.width * self.IMG_WT) + x1)
        y2 = int((ele_bbox.height * self.IMG_HT) + y1)
        face_bboxes.append([x1,y1,x2,y2])
      self.face_count = len(face_bboxes)

    else:
      self.face_count = 0
    return face_bboxes


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

    self.thumbs_down_count = 0
    self.down_count_done_flag = False
    self.fin_flag = False


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
          if not self.up_count_done_flag:
            self.thumbs_up_count += 1
            self.up_count_done_flag = True
        elif thumb_tip_val.y > index_tip_val.y > middle_tip_val.y > ring_tip_val.y > pinky_tip_val.y:
          if not self.down_count_done_flag:
            self.thumbs_down_count += 1 
            self.down_count_done_flag = True
    else:
      self.thumbs_up_count = 0
      self.thumbs_down_count = 0
      self.up_count_done_flag = False
      self.down_count_done_flag = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='To Run App >> python count_face_hand.py -v <path-to-video-file> ')
    parser.add_argument('-v', '--videosrc', default=0 ,help="Pass the path to input video file, by default reads webcam source of the PC")
    args = parser.parse_args()

    #Test class on Video
    vidcap = cv2.VideoCapture(args.videosrc)
   
    facedetvid = FaceDetector()
    handgestvid = HandGestureDetector(maxHands=1)

    total_count_up = 0
    total_count_down = 0

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
      f_bbox = facedetvid.getDetectedFaces(frame)
      handgestvid.checkThumbsUpDown(frame)


      for ele in f_bbox:
        cv2.rectangle(frame,(ele[0],ele[1]),(ele[2],ele[3]),(128,255,0),2)
      cv2.putText(frame, "FaceCount = " +str(facedetvid.face_count), (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255), 2)
      cv2.putText(frame, "ThumbsUpCount = " +str(handgestvid.thumbs_up_count), (10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255), 2)
      cv2.putText(frame, "ThumbsDownCount = " +str(handgestvid.thumbs_down_count), (10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255), 2)
      cv2.imshow('Output', frame)
      
      if cv2.waitKey(1) == ord('q'):
          break

    vidcap.release()
    cv2.destroyAllWindows()
    print("Total Up = ", total_count_up)
    print("Total Down = ", total_count_down)

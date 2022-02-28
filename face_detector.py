import cv2
import mediapipe as mp
import sys


class FaceDetector:
  """ 
  Detect faces in the given image/video frame and extract bonding box information.
  """
  def __init__(self, minConfScore=0.55) -> None:
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='To Run App >> python count_face_hand.py -v <path-to-video-file> ')
    parser.add_argument('-v', '--videosrc', default=0 ,help="Pass the path to input video file, by default reads webcam source of the PC")
    args = parser.parse_args()

    #Test class on Video
    vidcap = cv2.VideoCapture(args.videosrc)
   
    facedetvid = FaceDetector()


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


      for ele in f_bbox:
        cv2.rectangle(frame,(ele[0],ele[1]),(ele[2],ele[3]),(128,255,0),2)
      cv2.putText(frame, "FaceCount = " +str(facedetvid.face_count), (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255), 2)
      cv2.imshow('Output', frame)
      
      if cv2.waitKey(1) == ord('q'):
          break

    vidcap.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np

out = cv2.VideoWriter('./syf_dream_20170731_153920_us.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') , 20, (96,96))

for i in range(0,150):
        # get a frame
    frame = cv2.imread('./syf_dream_20170731_153920_us/%d.jpg' % i)
        # save a frame
    out.write(frame)
        # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
out.release()
cv2.destroyAllWindows()
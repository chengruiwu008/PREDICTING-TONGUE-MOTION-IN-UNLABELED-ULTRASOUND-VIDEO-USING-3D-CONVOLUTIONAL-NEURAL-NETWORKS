import numpy as np
import cv2

dir = 'cut at 47_48 snake_line'
cap = cv2.VideoCapture('E:/for the paper/'+dir+'.avi')
i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (96, 96))
    cv2.imwrite('E:/for the paper/'+dir+'/' + str(i) + '.jpg',frame)
    gray = cv2.cvtColor(frame, 0)#cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1
cap.release()
cv2.destroyAllWindows()
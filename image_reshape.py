import cv2
import numpy as np
import linecache

for i in range(1,354):
    # dir = linecache.getline('./list/Bruce_list_folder_2341.txt', (i+1))
    # dir = dir.strip('\n')
    image = cv2.imread('./Bruce_list01/Bruce_list1_sent50/image (%d).bmp'% i,0)
    image_96 = cv2.resize(image,(96,96))
    cv2.imwrite('./Bruce_list_1_sentence_50_9696/' + str(i-1) + '.jpg', image_96)
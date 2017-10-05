import cv2
import numpy as np
import matplotlib.pyplot as plt

def loss(img_1,img_2):
    sum_ = 0.0
    for k in range(96*96):
        # for j in range(96):
        sum_ = sum_ + (img_1[k]-img_2[k])**2
    return sum_/(96*96)

# def loss(img_1,img_2,alpha = 2):
#     sum_0 = 0
#     for i in range(96):
#         for j in range(95):
#             sum_0 = sum_0 + abs(abs(img_1[i,j]-img_1[i,j+1])-abs(img_2[i,j]-img_2[i,j+1]))**alpha
#     sum_1 = 0
#     for i in range(95):
#         for j in range(96):
#             sum_1 = sum_1 + abs(abs(img_1[i,j]-img_1[i+1,j])-abs(img_2[i,j]-img_2[i+1,j]))**alpha
#     # print(sum_0+sum_1)
#     return sum_0+sum_1

test_num = [400, 314, 355, 390, 296, 321, 373, 398, 373, 447, 383, 301, 386, 273, 345, 218, 337, 266, 387, 455, 405, 454,
       444, 401, 453, 443, 286, 249, 181, 410, 446, 449, 419, 441, 445, 345, 398, 407, 291, 449, 362, 470, 232, 416,
       138, 410, 268, 233, 372, 453, 398, 302, 323, 357, 277, 332, 438, 333, 363, 357, 462, 315, 173, 402, 429, 313,
       324, 276, 330, 377, 383, 368, 221, 423, 254, 297, 236, 416, 297, 379, 324, 368, 182, 390, 343, 449, 423, 457,
       425, 268, 209, 127, 331, 317, 282, 426, 427, 390, 444, 414]

# n = 68
# kaishi = 256
# jieshu = kaishi + 250
# dir = './WSJ0_test_data/reshape_96_96/test_'+str(j)+'/'
# dir_p = './WSJ0_test_data/test_result_3d_CNN/test_'+str(j)+'/'

def compare_image(dir,dir_p,kaishi):
    loss_0 = []
    loss_1 = []
    loss_2 = []
    loss_3 = []
    loss_4 = []
    loss_5 = []
    loss_6 = []
    loss_7 = []
    loss_aver = []
    loss_tar_real_pred = []
    loss_tar_liner = []
    s = 0
    # for i in range(0, (test_num[j]-9)):
    jieshu = kaishi + 250
    for i in range(kaishi,jieshu):
        image_0 = cv2.imread(dir + '/'+ str(i) + '.jpg',0)
        image_0 = np.array(image_0, dtype='int').reshape(-1)
        image_1 = cv2.imread(dir + '/'+str(i + 1) + '.jpg',0)
        image_1 = np.array(image_1, dtype='int').reshape(-1)
        image_2 = cv2.imread(dir + '/'+str(i + 2) + '.jpg',0)
        image_2 = np.array(image_2, dtype='int').reshape(-1)
        image_3 = cv2.imread(dir + '/'+str(i + 3) + '.jpg',0)
        image_3 = np.array(image_3, dtype='int').reshape(-1)
        image_4 = cv2.imread(dir + '/'+str(i + 4) + '.jpg',0)
        image_4 = np.array(image_4, dtype='int').reshape(-1)
        image_5 = cv2.imread(dir + '/'+str(i + 5) + '.jpg',0)
        image_5 = np.array(image_5, dtype='int').reshape(-1)
        image_6 = cv2.imread(dir + '/'+str(i + 6) + '.jpg',0)
        image_6 = np.array(image_6, dtype='int').reshape(-1)
        image_7 = cv2.imread(dir + '/'+str(i + 7) + '.jpg',0)
        image_7 = np.array(image_7, dtype='int').reshape(-1)
        image_tar = cv2.imread(dir + '/%d.jpg' % (i + 8),0)
        image_tar = np.array(image_tar, dtype='int').reshape(-1)
        image_pred_ = cv2.imread(dir_p + '/'+ str(i + 8) + '.jpg',0)
        image_pred_ = np.array(image_pred_, dtype='int').reshape(-1)
        image_average = (image_0 + image_1 + image_2 + image_3 + image_4 + image_5 + image_6 + image_7) / 8
        image_average = np.array(image_average, dtype='int').reshape(-1)
        image_liner = (image_0*0.05987975 + image_1*0.01374708 + image_2*0.00563895 + image_3*0.00163824
                       - image_4*0.0032593 + image_5*0.00613215 + image_6*0.11424943 + image_7*0.80009717)
        # [0.05987975, 0.01374708, 0.00563895, 0.00163824, -0.0032593, 0.00613215, 0.11424943, 0.80009717]
        image_liner = np.array(image_liner, dtype='int').reshape(-1)

        image_pred = image_tar

        loss_0.append(loss(image_pred, image_0))
        loss_1.append(loss(image_pred, image_1))
        loss_2.append(loss(image_pred, image_2))
        loss_3.append(loss(image_pred, image_3))
        loss_4.append(loss(image_pred, image_4))
        loss_5.append(loss(image_pred, image_5))
        loss_6.append(loss(image_pred, image_6))
        loss_7.append(loss(image_pred, image_7))
        loss_aver.append(loss(image_pred, image_average))
        loss_tar_real_pred.append(loss(image_tar, image_pred_))
        loss_tar_liner.append(loss(image_tar, image_liner))
        s += 1
        print('sum = ',s)
    return loss_7, loss_tar_real_pred, loss_tar_liner, loss_aver

# print(s)
# print('sum(loss_0)   ',sum(loss_0)/s)
# print('sum(loss_1)   ',sum(loss_1)/s)
# print('sum(loss_2)   ',sum(loss_2)/s)
# print('sum(loss_3)   ',sum(loss_3)/s)
# print('sum(loss_4)   ',sum(loss_4)/s)
# print('sum(loss_5)   ',sum(loss_5)/s)
# print('sum(loss_6)   ',sum(loss_6)/s)
# print('sum(loss_7)   ',sum(loss_7)/s)
# print('sum(loss_tar_pred)',sum(loss_tar_real_pred)/s)
# print('sum(loss_aver)',sum(loss_aver)/s)
# print('sum(loss_tar_liner)',sum(loss_tar_liner)/s)

# plt.figure('Snake test image from %d to %d'% (kaishi,jieshu,) + ' MSE')
# plt.plot(loss_0, '-.b',label='loss_0_tar')
# plt.plot(loss_1, '-.c',label='loss_1_tar')
# plt.plot(loss_2, '-.g',label='loss_2_tar')
# plt.plot(loss_3, '-.k',label='loss_3_tar')
# plt.plot(loss_4, '-.m',label='loss_4_tar')
# plt.plot(loss_5, '-.r',label='loss_5_tar')

def plot_3_image():

    dir_p = r'F:\CR.W\WSJ0_test_data\test_result_3d_CNN\test_67'
    dir = r'F:\CR.W\WSJ0_test_data\reshape_96_96\test_67'
    loss_7, loss_tar_real_pred, loss_tar_liner, loss_aver = compare_image(dir,dir_p,0)
    plt.figure('All 3 in one plot')
    frame1 = plt.subplot(311)
    # plt.plot(loss_6, '-.y', label='loss_6_tar')
    plt.plot(loss_7, '-k', label='8th image')
    plt.plot(loss_tar_liner, '-b', label='Linear')
    plt.plot(loss_aver, '-g',label='Average')
    plt.plot(loss_tar_real_pred, '-r', label='3DCNN')
    # plt.xlabel('number of images')
    plt.ylabel("WSJ0")
    frame1.axes.get_xaxis().set_visible(False)
    # frame1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    # plt.title('Snake test image from %d to %d'% (kaishi,jieshu,) + ' MSE')
    plt.legend(loc='best')

    dir_p = r'F:\CR.W\3d_CNN\3d_CNN_pred_us_real'
    dir = r'F:\CR.W\syf_dream_20170731_153920_us'
    loss_7, loss_tar_real_pred, loss_tar_liner, loss_aver = compare_image(dir, dir_p, 3550)

    frame2 = plt.subplot(312)
    plt.plot(loss_tar_liner, '-b', label='Linear')
    plt.plot(loss_7, '-k', label='8th image')
    plt.plot(loss_tar_real_pred, '-r', label='3DCNN')
    frame2.axes.get_xaxis().set_visible(False)
    # frame2.yaxis.get_major_formatter().set_powerlimits((0, 1))
    # plt.plot(loss_aver_real_pred_, '--k', label='prediction_average')
    # plt.plot(loss_tar_real_pred_, '--r', label='prediction_target')
    # plt.plot(loss_tar_average_, label='target-average')
    # plt.xlabel('number of images')
    plt.ylabel('TJU')

    dir_p = r'F:\CR.W\3d_CNN\3d_CNN_pred_snake'
    dir = r'F:\CR.W\syf_dream_20170731_153920_snake'
    loss_7, loss_tar_real_pred, loss_tar_liner, loss_aver = compare_image(dir, dir_p, 256)

    frame3 = plt.subplot(313)
    plt.plot(loss_tar_liner, '-b', label='Linear')
    plt.plot(loss_7, '-k', label='8th image')
    plt.plot(loss_tar_real_pred, '-r', label='3DCNN')
    frame3.yaxis.get_major_formatter().set_powerlimits((0, 1))

    plt.ylabel('Cross')
    # plt.xlabel('number of images')

    plt.show()

plot_3_image()

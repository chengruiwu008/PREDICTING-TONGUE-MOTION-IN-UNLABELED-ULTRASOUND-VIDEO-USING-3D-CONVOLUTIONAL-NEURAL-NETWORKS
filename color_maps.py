import numpy as np
import cv2
import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from math import sqrt

# k = 184

test_num = [400, 314, 355, 390, 296, 321, 373, 398, 373, 447, 383, 301, 386, 273, 345, 218, 337, 266, 387, 455, 405, 454,
       444, 401, 453, 443, 286, 249, 181, 410, 446, 449, 419, 441, 445, 345, 398, 407, 291, 449, 362, 470, 232, 416,
       138, 410, 268, 233, 372, 453, 398, 302, 323, 357, 277, 332, 438, 333, 363, 357, 462, 315, 173, 402, 429, 313,
       324, 276, 330, 377, 383, 368, 221, 423, 254, 297, 236, 416, 297, 379, 324, 368, 182, 390, 343, 449, 423, 457,
       425, 268, 209, 127, 331, 317, 282, 426, 427, 390, 444, 414]
ls=0

for j in range(0,100):

    dir = './WSJ0_test_data/reshape_96_96/test_'+str(j)+'/'
    dir_p = './WSJ0_test_data/test_result_3d_CNN/test_'+str(j)+'/'

    for i in range(0, (test_num[j]-9)):
        image_0 = cv2.imread(dir + str(i) + '.jpg',0)
        image_0 = np.array(image_0, dtype='int')#.reshape(-1)
        image_1 = cv2.imread(dir + str(i + 1) + '.jpg',0)
        image_1 = np.array(image_1, dtype='int')#.reshape(-1)
        image_2 = cv2.imread(dir + str(i + 2) + '.jpg',0)
        image_2 = np.array(image_2, dtype='int')#.reshape(-1)
        image_3 = cv2.imread(dir + str(i + 3) + '.jpg',0)
        image_3 = np.array(image_3, dtype='int')#.reshape(-1)
        image_4 = cv2.imread(dir + str(i + 4) + '.jpg',0)
        image_4 = np.array(image_4, dtype='int')#.reshape(-1)
        image_5 = cv2.imread(dir + str(i + 5) + '.jpg',0)
        image_5 = np.array(image_5, dtype='int')#.reshape(-1)
        image_6 = cv2.imread(dir + str(i + 6) + '.jpg',0)
        image_6 = np.array(image_6, dtype='int')#.reshape(-1)
        image_7 = cv2.imread(dir + str(i + 7) + '.jpg',0)
        image_7 = np.array(image_7, dtype='int')#.reshape(-1)
        image_tar = cv2.imread(dir + '%d.jpg' % (i + 8),0)
        image_tar = np.array(image_tar, dtype='int')#.reshape(-1)
        image_pred_ = cv2.imread(dir_p + str(i + 8) + '.jpg',0)
        image_pred_ = np.array(image_pred_, dtype='int')#.reshape(-1)
        image_average = (image_0 + image_1 + image_2 + image_3 + image_4 + image_5 + image_6 + image_7) * 0.125
        image_average = np.array(image_average, dtype='int')#.reshape(-1)
        image_linear = (image_0 * 0.05987975 + image_1 * 0.01374708 + image_2 * 0.00563895 + image_3 * 0.00163824
                       - image_4 * 0.0032593 + image_5 * 0.00613215 + image_6 * 0.11424943 + image_7 * 0.80009717)
        # [0.05987975, 0.01374708, 0.00563895, 0.00163824, -0.0032593, 0.00613215, 0.11424943, 0.80009717]
        image_linear = np.array(image_linear, dtype='int')#.reshape(-1)
        # target_img = cv2.imread(dir + str(k) + '.jpg', 0)
        # target_img = np.array(target_img, dtype='float')  # .reshape(-1)
        # pred_img = cv2.imread(dir_p + str(k) + '.jpg', 0)
        # pred_img = np.array(pred_img, dtype='float')  # .reshape(-1)

        # pred_mul_tar = np.sqrt(target_img * pred_img)
        # pred_mul_tar = np.sqrt(image_tar * image_pred_)
        # pred_mul_tar = np.array(pred_mul_tar, dtype='int')

        pred_tar = np.sqrt(image_tar * image_pred_) - image_tar
        linear_pred_tar = np.sqrt(image_tar * image_linear) - image_tar

        # pred_tar = pred_img - target_img
        # cv2.imwrite('./WSJ0_test_data/test_result_3d_CNN_sqrt_all/test_%d/%d.jpg' % (j,k), pred_mul_tar)

        plt.figure(0)

        plt.subplot(121)
        # plt.title('Sqrt(Prediction*Target)\n - Target')
        plt.title('Prediction - Target')
        plt.imshow(pred_tar, cmap='seismic', vmin=-30, vmax=30)
        plt.colorbar(orientation='horizontal')

        plt.subplot(122)
        # plt.title('Sqrt(Linear_Pred*Target)\n - Target')
        plt.title('Last_in - Target')
        plt.imshow(linear_pred_tar, cmap='seismic', vmin=-30, vmax=30)
        plt.colorbar(orientation='horizontal')

        # plt.figure(0)
        # plt.colorbar(orientation='horizontal')
        plt.savefig('./WSJ0_test_data/test_result_3d_CNN_color_map_varience_compare/%d.jpg' % ls)

        plt.close('all')
        print('%d is done' % ls)
        ls+=1

    # plt.show()

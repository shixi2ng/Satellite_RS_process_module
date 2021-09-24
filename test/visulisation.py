import cv2
import numpy as np

# 图片路径
img = cv2.imread('LC08_BSZ.png')
a = []
b = []
image_scale_down = 0.5
x = int(img.shape[0] / image_scale_down)
y = int(img.shape[1] / image_scale_down)
image = cv2.resize(img, (x, y))


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        try:
            img_temp = cv2.imread('E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Landsat_inundated_curve\\bsz_overview\\overview_NDVI_' + str(int(np.fix(x / 2))) + '_' + str(int(np.fix(y / 2))) + '.png')
            img_temp = cv2.resize(img_temp, (int(0.3 * img_temp.shape[1]), int(0.3 * img_temp.shape[0])))
            cv2.moveWindow('img_new', 40, 30)
            cv2.imshow('img_new', img_temp)
        except:
            print('No figure')
        cv2.imshow("image", image)
        print(str(int(np.fix(y / 2))), str(int(np.fix(x / 2))))


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", image)
cv2.waitKey(0)
print(a[0], b[0])
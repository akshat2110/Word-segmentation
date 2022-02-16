import cv2
import os
import fnmatch
import numpy as np
from itertools import groupby
import statistics

output_path = "/Final Output(20 images)/"
folder_path = "/Line images/"

if not os.path.exists(output_path):
    os.mkdir(output_path)

list_of_files = fnmatch.filter(os.listdir(folder_path), '*.tif')
f = 1

for file in list_of_files:
    # image = cv2.imread(os.path.abspath(folder_path + file), 0)
    image = cv2.imread(folder_path + file, 0)
    rows, cols = image.shape

    ret1, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    copy2 = np.copy(th2)
    th = np.copy(th2)

    def counting(i, j, c):
        if c <= threshold_count and 0 <= (i - 1) <= rows and 0 <= (j - 1) <= cols and th[i - 1][j - 1] == 0:
            th[i - 1][j - 1] = 155
            c = c + 1
            i = i - 1
            j = j - 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 <= (i - 1) <= rows and 0 <= j <= cols and th[i - 1][j] == 0:
            th[i - 1][j] = 155
            c = c + 1
            i = i - 1
            j = j
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 <= (i - 1) <= rows and 0 <= (j + 1) <= cols and th[i - 1][j + 1] == 0:
            th[i - 1][j + 1] = 155
            c = c + 1
            i = i - 1
            j = j + 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 <= i <= rows and 0 <= (j - 1) <= cols and th[i][j - 1] == 0:
            th[i][j - 1] = 155
            c = c + 1
            i = i
            j = j - 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 <= i <= rows and 0 <= (j + 1) <= cols and th[i][j + 1] == 0:
            th[i][j + 1] = 155
            c = c + 1
            i = i
            j = j + 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 <= (i + 1) <= rows and 0 <= (j - 1) <= cols and th[i + 1][j - 1] == 0:
            th[i + 1][j - 1] = 155
            c = c + 1
            i = i + 1
            j = j - 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 <= (i + 1) <= rows and 0 <= j <= cols and th[i + 1][j] == 0:
            th[i + 1][j] = 155
            c = c + 1
            i = i + 1
            j = j + 1
            c = counting(i, j, c)
            return c

        elif c <= threshold_count and 0 <= (i + 1) <= rows and 0 <= (j + 1) <= cols and th[i + 1][j + 1] == 0:
            th[i + 1][j + 1] = 155
            c = c + 1
            i = i + 1
            j = j + 2
            c = counting(i, j, c)
            return c

        else:
            return c

    c = 0
    lst = []
    threshold_count = 12
    for i in range(rows):
        for j in range(cols):
            if th[i][j] == 0:
                c = counting(i, j, c)
                if 0 < c <= threshold_count:
                    lst.append(c)
            c = 0
    val = int(statistics.median(lst))
    print(f"{file} = {val}")
    if val % 2 == 0:
        val = val + 1

    thresh = cv2.GaussianBlur(thresh, (val, val), 0)

    col_f, col_b = 0, 0
    for i in range(cols):
        flag1 = 1
        for j in range(rows):
            if flag1 == 1:
                if th2[j][i] == 0:
                    flag1 = 0
                    col_f = i
                    break
        if flag1 == 1:
            for k in range(rows):
                copy2[k][i] = 155
        else:
            break

    for i in reversed(range(cols)):
        flag2 = 1
        for j in reversed(range(rows)):
            if flag2 == 1:
                if th2[j][i] == 0:
                    flag2 = 0
                    col_b = i
                    break
        if flag2 == 1:
            for k in range(rows):
                copy2[k][i] = 155
        else:
            break

    row_t, row_b = 0, 0
    flag1, flag2 = 1, 1
    for i in range(rows):
        for j in range(cols):
            if flag1 == 1:
                if th2[i][j] == 0:
                    flag1 = 0
                    row_t = i

            if flag2 == 1:
                if th2[rows - 1 - i][j] == 0:
                    flag2 = 0
                    row_b = rows - 1 - i
        if flag1 == 1:
            copy2[i] = 155
        if flag2 == 1:
            copy2[rows - 1 - i] = 155

    print(f"Col from front: {col_f}, Col from back: {col_b}")
    print(f"Row from top: {row_t}, Row from bottom: {row_b}")

    max = 0
    row_pos = 0
    ele = 255
    for i in range(row_t, row_b + 1):
        res = [list(j) for i, j in groupby(copy2[i][col_f:col_b + 1].tolist(), lambda x: x == ele) if not i]
        z = 0
        for j in res:
            z = z + len(j)
        if z > max:
            max = z
            row_pos = i

    length = []
    ele = 0
    res = [list(j) for i, j in groupby(copy2[row_pos][col_f:col_b + 1].tolist(), lambda x: x == ele) if not i]

    z = 0
    for i in res:
        length.append(len(i))
        z = z + len(i)

    median = int(statistics.median(length))
    print(f"Median: {median}")

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (median, median))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img2 = image.copy()
    list1, list2 = [], []
    k = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        list1.append(x)

        crop = img2[y:y+h, x:x+w]
        list2.append(crop)

    zipped_lists = zip(list1, list2)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list1 = [element for _, element in sorted_zipped_lists]

    p = output_path + "Crop_{}/".format(file)
    if not os.path.exists(p):
        os.mkdir(p)
    x += 1

    print(p)

    for crop_img in sorted_list1:
        cv2.imwrite(os.path.join(p, '{}.tif'.format(f)), crop_img)
        f += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()

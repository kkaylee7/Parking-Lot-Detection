import cv2
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report

def out_file(df, title):
    os.makedirs('out_file/', exist_ok=True)
    df.to_csv('out_file/' + str(video.split("/")[-1].split(".")[0]) + "_" + title + '.csv')

def parking_space():
    cap = cv2.VideoCapture(video)
    points = []

    def draw_parking_spot(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            for i, pts in enumerate(points):
                x1, y1 = pts
                if x1 < x + w and y1 < y + h:
                    points.pop(i)

    w, h = 50, 30
    while True:
        success, frame = cap.read()
        if success == False:
            break

        for pts in points:
            cv2.rectangle(frame, pts, (pts[0] + w, pts[1] + h), (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        cv2.setMouseCallback("Frame", draw_parking_spot)
        if cv2.waitKey(32) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    points = np.array(points)
    points = points.reshape(-1, 2)
    df = pd.DataFrame(points, columns=['x1', 'y1'])
    df['x2'] = df['x1'] + w
    df['y2'] = df['y1'] + h
    print(df)
    out_file(df, "parking_lot")


def parking_detection():
    df = pd.read_csv("out_file/" + str(video.split("/")[-1].split(".")[0]) + "_parking_lot.csv")
    lot_position = df.drop(columns=df.columns[0])
    print(lot_position.head())

    label = []

    def checkParkingSpace(df, frame):
        for ind in df.index:
            x1, x2, y1, y2 = df['x1'][ind], df['x2'][ind], df['y1'][ind], df['y2'][ind]
            imgCrop = frame[y1: y2, x1: x2]

            # apply edge detection and number of pixels in image
            gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edge = cv2.Canny(blurred, 50, 255)
            count = cv2.countNonZero(edge)
            if count > 150:
                # space is occupied
                red = (0, 0, 255)
                result = "occupied"
                label.append(result)
                cv2.rectangle(frame, (x1, y1), (x2, y2), red, 3)

            else:
                # space is empty
                green = (0, 255, 0)
                result = "empty"
                label.append(result)
                cv2.rectangle(frame, (x1, y1), (x2, y2), green, 3)

        return df

    cap = cv2.VideoCapture(video)

    current_frame = 0
    train = pd.DataFrame()
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        lot_position = checkParkingSpace(lot_position, frame)
        train = pd.concat([train, lot_position])

        cv2.imshow("Frame", frame)
        if cv2.waitKey(32) & 0xFF == ord("q"):
            break
        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    train["class"] = label
    out_file(train, "train")

def overlap(boxA, boxB):
    if boxA[1] < boxB[3] or boxA[0] < boxB[2]:
        # no overlap
        return "empty"
    elif boxA[2] > boxB[0] or boxA[3] > boxB[2]:
        # no overlap
        return "empty"
    else:
        # overlap
        return "occupied"


def checkinside(box, p):
   logic1 = (p[0] > box[0] and p[0] < box[2] and p[1] > box[1] and p[1] < box[3])
   logic2 = (p[2] > box[0] and p[2] < box[2] and p[3] > box[1] and p[3] < box[3])
   if logic1 or logic2 == True:
       return "occupied"
   else:
       return "empty"

label = []

def training():
    def checkOverlap(df, frame, frame_count, pd_objects):
        # for each unique parking spot, check if any motion is detected in each spot
        id = 0
        for ind in df.index:
            x1, x2, y1, y2 = df['x1'][ind], df['x2'][ind], df['y1'][ind], df['y2'][ind]
            # assume its empty at first
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            result = "empty"

            # check if there is a moving object in the current frame
            exist = frame_count in pd_objects.current_frame
            if exist:
                mask = pd_objects['current_frame'].values == current_frame
                current_object = pd_objects[mask]
                # if object is a person then disregard
                current_object = current_object[current_object.object_type != 1]

                # detect all the objects in that frame
                for ind2 in current_object.index:
                    x11, y11, w, h = current_object['object_bbox_lefttop_x'][ind2], current_object['object_bbox_lefttop_y'][
                        ind2], current_object['object_bbox_width'][ind2], current_object['object_bbox_height'][ind2]
                    x22 = x11 + w
                    y22 = y11 + h
                    cv2.rectangle(frame, (x11, y11), (x22, y22), (255, 0, 0), 3)  # objects
                    parking_box = np.array([x1, y1, x2, y2])
                    object_box = np.array([x11, y11, x22, y22])
                    overlap_check = overlap(parking_box, object_box)
                    AinB = checkinside(parking_box, object_box)
                    BinA = checkinside(object_box, parking_box)
                    if overlap_check == "occupied" or AinB == "occupied" or BinA == "occupied":
                        result = "occupied"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        # if the parking lot is occupied we can break since we already know there is an object there
                        break

            label.append(result)
            id += 1


    df = pd.read_csv("out_file/" + str(video.split("/")[-1].split(".")[0]) + "_parking_lot.csv")
    lot_position = df.drop(columns=df.columns[0])
    object_path = "annotations/" + str(video.split("/")[-1].split(".")[0]) + ".viratdata.objects.txt"
    pd_objects = pd.read_csv(object_path, sep=" ", usecols=[0, 2, 3, 4, 5, 6, 7],
                             names=["object_ID", "current_frame", "object_bbox_lefttop_x",
                                    "object_bbox_lefttop_y", "object_bbox_width",
                                    "object_bbox_height", "object_type"])
    cap = cv2.VideoCapture(video)

    current_frame = 0
    test = pd.DataFrame()

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        checkOverlap(lot_position, frame, current_frame, pd_objects)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(32) & 0xFF == ord("q"):
            break
        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    test["class"] = label
    print(test)
    out_file(test, "test")


def test():
    df = pd.read_csv("out_file/" + str(video.split("/")[-1].split(".")[0]) + "_train.csv")
    train = df.drop(columns=df.columns[0])
    print(train.head())
    test = pd.read_csv("out_file/" + str(video.split("/")[-1].split(".")[0]) + "_test.csv")
    act = test.drop(columns=test.columns[0])
    print(test.head())
    pred = train["class"]
    print(classification_report(act, pred, labels=["empty", "occupied"], zero_division=1))

def run():
    parking_space()
    parking_detection()
    training()
    test()

video = "/Users/kkaylee7/Desktop/dataset/VIRAT_S_040103_08_001475_001512.mp4"
run()

video = "/Users/kkaylee7/Desktop/dataset/VIRAT_S_040000_00_000063_000085.mp4"
run()

video = "/Users/kkaylee7/Desktop/dataset/VIRAT_S_050000_08_001235_001295.mp4"
run()

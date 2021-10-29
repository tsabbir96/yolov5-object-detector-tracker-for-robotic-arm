import argparse
import warnings
warnings.filterwarnings("ignore")
from utils.datasets import *
from utils.utils import *

import cv2
from math import atan2, cos, sin, sqrt, pi
from plyfile import PlyData, PlyElement
import numpy as np

# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objectCoors = OrderedDict()
        self.disappeared = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.objectCoors[self.nextObjectID] = rect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.objectCoors[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = np.zeros((len(rects), 4), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects[i] = (startX, startY, endX, endY)
   
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            objectCoors =  list(self.objectCoors.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.objectCoors[objectID] = inputRects[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            #compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
   
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])
        # return the set of trackable objects
        return self.objects, self.objectCoors

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)
    
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]
 
def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]
    
    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)
    
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]
    
    # Label with the rotation angle
    label = str(-int(np.rad2deg(angle)) - 90)
    # textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    
    return label

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    return np.stack([x,y,z], axis=1)

def getTopLeft(pts):
    return np.asarray([np.amin(pts[:,0]), np.amin(pts[:,1])])

def align_to_origin(topLeft, coors):
    coors[:, 0] = coors[:, 0] - topLeft[0]
    coors[:, 1] = coors[:, 1] - topLeft[1]
    return coors

def point_in_circle(centre, point, r):
  d = math.sqrt((centre[0] - point[0])**2  +  (centre[0] - point[0])**2)
  if d < r:
    return True
  else:
    return False

def transform_point_to_3D(point, origin3D, scaleRatio):
    x3D = origin3D[0] + point[0] / scaleRatio[0]
    y3D = origin3D[1] + point[1] / scaleRatio[1]
    return [x3D, y3D]

def get_mean_depth(coors, point3D, r):
    depths = []
    for each_point in coors:
      if(point_in_circle(point3D, each_point, r)):
        depth = each_point[2]
        depths.append(depth)
    mean_depth = np.mean(depths)
    return mean_depth

def detect(save_img=False):

    out, source, weights, half, view_img, save_txt, imgsz, detect_angle = \
        opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt, opt.img_size, opt.detect_angle
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    ct = CentroidTracker()
  
    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model']
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[0,255,0], [255,0,0]]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        first_object = True
        
        ply_coors =  read_ply("/content/yolov5-single-object-detector/sample_img/3D/" + path.split('/')[-1].split('.')[0] + ".ply")
        ply_coors = ply_coors.astype(int)

        origin3D =  getTopLeft(ply_coors)
        ply_coors = align_to_origin(origin3D, ply_coors)

        scaleRatio = [5.7, 5.7]

        
        
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                conf_scores = []
                # Write results
                rects = []
                
                for *xyxy, conf, cls in det:
                  rects.append(xyxy)
                  # cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                
                objects, objectCoors = ct.update(rects)

                objectIDs = []
                coors = []
                for (objectID, coor) in objectCoors.items():
                  text = "ID {}".format(objectID)
                  objectIDs.append(objectID)
                  coors.append(coor)
                  
                objectID = min(objectIDs)
                min_idx = objectIDs.index(objectID)
                coor = coors[min_idx] 
                
                if 0 < ct.disappeared[objectID] < 50:
                  # cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                  # cv2.circle(im0, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)
                  cv2.rectangle(im0, (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3])), (255, 0, 0), 2)
              
                else:
                  #cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                  # cv2.circle(im0, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                  
                  if detect_angle:  #detect rotaiton angle
                      x,y,w,h=int(coor[0]), int(coor[1]), int(coor[2] - coor[0]), int(coor[3] - coor[1])
                      crop_img = im0[y:y+ h, x:x + w]
                      processed_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                      ret, processed_img = cv2.threshold(processed_img, 15, 255, cv2.THRESH_BINARY_INV)
                      processed_img = cv2.dilate(processed_img.copy(), None, iterations=1)
                      contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                      all_area = []
                      for cntr_id, c in enumerate(contours):
                        area = cv2.contourArea(c)
                        all_area.append(area)
                      if len(all_area) == 0:
                        continue
                      max_value = max(all_area)
                      max_idx = all_area.index(max_value)
                      angle = getOrientation(contours[max_idx], crop_img)
                      im0[y:y+ h, x:x + w] = crop_img

                      cx = int((coor[0] + coor[2]) / 2.0)
                      cy = int((coor[1] + coor[3]) / 2.0)

                      point2D = [cx, cy]

                      origin3D = [0, 0]
                      point3D = transform_point_to_3D(point2D, origin3D, scaleRatio)
                      mean_depth = get_mean_depth(ply_coors, point3D, 50)

                      print("Central Point (x,y) = ({},{})  Mean Depth = {}  Rotation Angle = {}".format(cx,cy,mean_depth,angle))
                  
                  cv2.rectangle(im0, (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3])), (0, 255, 0), 2)


                  # if save_txt:  # Write to file
                  #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                  #     with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                  #         file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                  # if first_object:  # track single object
                  #     #label = '%s %.2f' % (names[int(cls)], conf)
                  #     #plot_one_box(xyxy, im0, label=label, color=colors[1], line_thickness=3)
                  #     first_object = False
                  #     continue

                  # if save_img or view_img:  # Add bbox to image
                  #     label = '%s %.2f' % (names[int(cls)], conf)
                  #     plot_one_box(xyxy, im0, label=label, color=colors[0], line_thickness=3)

                    

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        # print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    # print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--detect-angle', action='store_true', help='detect rotation angle')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # print(opt)

    with torch.no_grad():
        detect()

import cv2
import numpy as np
import pandas as pd
from detectionTracking import Tracking
from actionClassification import ActionClassifier
from engagementClassification import EngagementClassifier
import math
import utils

cv2.namedWindow('Student Engagement Measurement Demo', cv2.WINDOW_NORMAL)
from engageConfig import *

if __name__ == '__main__':

    st_id =0

    classes_all = pd.read_csv(labels_file, usecols=['id', 'name'])
    action_names = [class_des for i, class_des in classes_all.values.tolist()]

    actionClassifier = ActionClassifier(action_names)
    engagementClassifier = EngagementClassifier(action_names, usedLLM='llama', contextAware=False)

    vid = cv2.VideoCapture(v_path)

    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frm_count = 0
    sts_trackers_data = {}
    sts_predActionsSeq = {}
    sts_engagement_levels_dict = {}
    frms = []
    displayFrms = []
    pred_action =""
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        display_frame = frame.copy()
        frm_count += 1
        frms.append(frame.copy())
        # ===recognize actions every 3 seconds ===#
        if frm_count >= clip_len and frm_count % clip_len == 0:
            print(f"=================****************************==============")

            inds = actionClassifier.uniformSampling(len(frms))
            sampledSubFrms = [frms[i] for i in inds]
            try:
                pred_action = actionClassifier.classify(sampledSubFrms)
                print(f" Id: {st_id} Action: {pred_action}")

                if st_id in sts_predActionsSeq:
                    sts_predActionsSeq[st_id].append(pred_action)
                else:
                    sts_predActionsSeq[st_id] = [pred_action]

            except Exception as e:
                print(f"Error: {e}")
            frms.clear()

        # ===engagement classification===#
        if frm_count % 1800 == 0:
            print(f"////////****************************/////////")
            # ====filter sts_predActionsSeq===#
            try:
                sts_engagement_levels_dict = engagementClassifier.classify_action_sequence(sts_predActionsSeq)
            except Exception as e:
                print(f"Error: {e}")
            sts_predActionsSeq.clear()


        display_frame = cv2.putText(display_frame, f"Pred Act: {pred_action}", (5, height-20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 30), 3, )
        if st_id in sts_engagement_levels_dict:
            display_frame = cv2.putText(display_frame, f"{sts_engagement_levels_dict[st_id]}",
                                        (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250),3,)

        cv2.imshow('Student Engagement Measurement Demo', display_frame)
        # cv2.namedWindow('Frms', cv2.WINDOW_NORMAL)
        # cv2.imshow('Frms', frms[-1])

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Release resources
    vid.release()
    cv2.destroyAllWindows()

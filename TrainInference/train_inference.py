

import scipy
import torch
import numpy as np
import pandas as pd
def train_net(
        model, 
        train_loader,
        train_optimizer,
        train_loss,
        device='cpu',
        t2_only=False,
        t2_dwi=False,
        no_rad=False
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.train()
    
    _step = 0.
    _loss = 0.
    for batch in train_loader:

        img, label = batch["img"].to(device), batch["label"].to(device)
        if t2_only:
            img = img[:, 0, ...].unsqueeze(1)
        elif t2_dwi:
            img = img[:, [0, 2], ...]

        
        if not no_rad:
            tumor = (label != 0).float()
            img = torch.cat([
                img,
                tumor
            ], dim=1)


        # forward pass of selected data
        output = model(img)
        
        loss = train_loss(output, label)

        loss.backward()
        train_optimizer.step()
        train_optimizer.zero_grad()

        _loss += loss.item()
        _step += 1.
    _epoch_loss = _loss / _step

    return _epoch_loss


def count_unique_contours(segmentation_map: torch.Tensor):
    """
    Counts the number of unique contours in a PyTorch segmentation map.
    
    Args:
        segmentation_map (torch.Tensor): A 2D tensor representing the segmentation map (H, W).
    
    Returns:
        int: Number of unique contours.
        torch.tensor: indexes
    """
    # Convert to NumPy for processing
    seg_np = segmentation_map.cpu().numpy().astype(int)

    # Find connected components (excluding background, label=0)
    labeled_array, num_features = scipy.ndimage.label(seg_np)

    return [
        torch.tensor(np.column_stack(np.where(labeled_array == i + 1)))
        for i in range(num_features)
    ], num_features

    
    



def post_process(x: torch.tensor):
    
    x = torch.argmax(x, dim=1)
    return x.to(float)


def inference_post_process(o):
    o = post_process(o) # make sure all value ins between 0-1
    return o # just get the 

    
# def inference_net(
#         model, 
#         inference_loader,
#         th=0.5,
#         device='cpu',
#     ):
#     # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
#     model.to(device)
#     model.eval()
    
    
#     TP_gt, FP_gt = 0, 0
#     TP_pred, FP_pred = 0, 0

#     for batch in inference_loader:
#         # print(batch["patient_name"], batch["patient_uid"])
#         img, label = batch["img"].to(device), batch["label"].to(device)
#         tumor = (label != 0).float()
#         img = torch.cat([
#             img,
#             tumor
#         ], dim=1)
#         contour_list, _ = count_unique_contours(label[0, 0, ...])
        
        
#         # forward pass 
#         output = model(img)
#         output = inference_post_process(output)[0]
        
        
        
        
        
#         for contour_index in contour_list:
            
#             gt = label[0, 0, ...][contour_index[:, 0], contour_index[:, 1], contour_index[:, 2]]
#             gt = torch.unique(gt).item()

#             if gt == 1:
#                 TP_gt += 1
#             else:
#                 FP_gt += 1
            
            
#             num_voxcels = contour_index.shape[0]
#             prediction = output[contour_index[:, 0], contour_index[:, 1], contour_index[:, 2]]
            
#             num_TP_voxcels = torch.count_nonzero((prediction == 1).int())
            
            
            
            
#             prediction = 1 if (num_TP_voxcels / num_voxcels) > 0.5 else 2
            

#             if prediction == 1 and gt == 1:
#                 TP_pred += 1
#             if prediction == 2 and gt == 2:
#                 FP_pred += 1

    
#     # print(TP_gt, FP_gt)
#     # print(TP_pred, FP_pred)
#     return TP_pred/TP_gt, FP_pred/FP_gt

@torch.no_grad()
def inference_net_lesion(
        model,
        inference_loader,
        th=0.5,
        device='cpu',
        pirads_sheet = None,
        t2_only=False,
        t2_dwi=False,
        no_rad=False
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.eval()
    num_examples = 0
    num_tumor = 0
    P, TP, N, TN, FP, FN = 0, 0, 0, 0, 0, 0
    TP_R_3, TN_R_3, FP_R_3, FN_R_3 = 0, 0, 0, 0
    TP_R_4, TN_R_4, FP_R_4, FN_R_4 = 0, 0, 0, 0
    TP_R_5, TN_R_5, FP_R_5, FN_R_5 = 0, 0, 0, 0
    
    
    P_Pat, N_Pat = 0, 0
    TP_Pat, TN_Pat, FP_Pat, FN_Pat = 0, 0, 0, 0
    
    TP_Pat_R_3, TN_Pat_R_3, FP_Pat_R_3, FN_Pat_R_3 = 0, 0, 0, 0
    TP_Pat_R_4, TN_Pat_R_4, FP_Pat_R_4, FN_Pat_R_4 = 0, 0, 0, 0
    TP_Pat_R_5, TN_Pat_R_5, FP_Pat_R_5, FN_Pat_R_5 = 0, 0, 0, 0
    
    
    priads_sheet = pd.read_excel(pirads_sheet)
    for batch in inference_loader:
        img, label = batch["img"].to(device), batch["label"].to(device)
        if t2_only:
            img = img[:, 0, ...].unsqueeze(1)
        elif t2_dwi:
            img = img[:, [0, 2], ...]


        if not no_rad:
            tumor = (label != 0).float()
            img = torch.cat([
                img,
                tumor
            ], dim=1)
        output = model(img)
        output = inference_post_process(output)
        
        
        
        position_list, num_unique = count_unique_contours(label)
        patient_name, patient_uid = batch['patient_name'][0], batch['patient_uid'][0]


        filtered_df = priads_sheet[priads_sheet['Patient ID'] == patient_name]
        filtered_df = filtered_df[filtered_df['seriesInstanceUID_MR'] == patient_uid]
        filtered_df = filtered_df.drop_duplicates(subset='ROI Volume (cc)')
        
        
        volumn_list = filtered_df['ROI Volume (cc)'].tolist()
        pirads_list = filtered_df['UCLA Score (Similar to PIRADS v2)'].tolist()
        # Combine and sort by list_a
        combined = sorted(zip(volumn_list, pirads_list), reverse=True)  # ascending sort
        # Unzip back
        volumn_list, pirads_list = zip(*combined)
        position_list = sorted(position_list, key=lambda x: x.shape[0], reverse=True)
        
        # print(volumn_list, pirads_list)
        # print([x.shape[0] for x in position_list])
        
        
        if len(filtered_df) == num_unique:
            num_examples+=1
            num_tumor += num_unique
            
            patient_positive_ai,patient_positive_3,patient_positive_4,patient_positive_5=False,False,False,False
            for pirads, index in zip(pirads_list, position_list):
                # print(index.shape, output.shape)
                # num_voxcel = index.shape[0]
                pathlogy_gt = label[0, 0, index[:, 2], index[:, 3], index[:, 4]]
                pathlogy_gt = torch.unique(pathlogy_gt).item()
                if pathlogy_gt == 1:
                    P += 1
                else:
                    N += 1
                prediction = output[0, index[:, 2], index[:, 3], index[:, 4]]
                count_1 = (prediction == 1).sum().item()  # Count number of 1's
                count_2 = (prediction == 2).sum().item()  # Count number of 2's
                
                if count_1 > count_2 and pathlogy_gt == 1:
                    TP += 1
                elif count_1 < count_2 and pathlogy_gt == 2:
                    TN += 1
                if count_1 > count_2 and pathlogy_gt == 2:
                    FP += 1
                elif count_1 < count_2 and pathlogy_gt == 1:
                    FN += 1
                    
                
                if pirads >= 3 and pathlogy_gt == 1:
                    TP_R_3 += 1
                elif pirads < 3 and pathlogy_gt == 2:
                    TN_R_3 += 1
                elif pirads >= 3 and pathlogy_gt == 2:
                    FP_R_3 += 1
                elif pirads < 3 and pathlogy_gt == 1:
                    FN_R_3 += 1
                
                if pirads >= 4 and pathlogy_gt == 1:
                    TP_R_4 += 1
                elif pirads < 4 and pathlogy_gt == 2:
                    TN_R_4 += 1
                elif pirads >= 4 and pathlogy_gt == 2:
                    FP_R_4 += 1
                elif pirads < 4 and pathlogy_gt == 1:
                    FN_R_4 += 1
                    
                if pirads >= 5 and pathlogy_gt == 1:
                    TP_R_5 += 1
                elif pirads < 5 and pathlogy_gt == 2:
                    TN_R_5 += 1
                elif pirads >= 5 and pathlogy_gt == 2:
                    FP_R_5 += 1
                elif pirads < 5 and pathlogy_gt == 1:
                    FN_R_5 += 1
                    
                    
                patient_positive_gt, patient_positive_ai, patient_positive_3,\
                    patient_positive_4, patient_positive_5=False, False, False, False, False
                if pathlogy_gt == 1:
                    patient_positive_gt=True
                if count_1>count_2:
                    patient_positive_ai=True
                if pirads>=3:
                    patient_positive_3=True
                if pirads>=4:
                    patient_positive_4=True
                if pirads>=5:
                    patient_positive_5=True
            
            if patient_positive_gt == True:
                P_Pat +=1
            else:
                N_Pat += 1
            
            
            if patient_positive_ai == patient_positive_gt and patient_positive_gt == True:
                TP_Pat += 1
            elif patient_positive_ai == patient_positive_gt and patient_positive_gt == False:
                TN_Pat += 1
            elif patient_positive_ai != patient_positive_gt and patient_positive_gt == False:
                FP_Pat += 1
            elif patient_positive_ai != patient_positive_gt and patient_positive_gt == True:
                FN_Pat += 1
                
            if patient_positive_3 == patient_positive_gt and patient_positive_gt == True:
                TP_Pat_R_3 += 1
            elif patient_positive_3 == patient_positive_gt and patient_positive_gt == False:
                TN_Pat_R_3 += 1
            elif patient_positive_3 != patient_positive_gt and patient_positive_gt == False:
                FP_Pat_R_3 += 1
            elif patient_positive_3 != patient_positive_gt and patient_positive_gt == True:
                FN_Pat_R_3 += 1
                
                
            if patient_positive_4 == patient_positive_gt and patient_positive_gt == True:
                TP_Pat_R_4 += 1
            elif patient_positive_4 == patient_positive_gt and patient_positive_gt == False:
                TN_Pat_R_4 += 1
            elif patient_positive_4 != patient_positive_gt and patient_positive_gt == False:
                FP_Pat_R_4 += 1
            elif patient_positive_4 != patient_positive_gt and patient_positive_gt == True:
                FN_Pat_R_4 += 1

                
            if patient_positive_5 == patient_positive_gt and patient_positive_gt == True:
                TP_Pat_R_5 += 1
            elif patient_positive_5 == patient_positive_gt and patient_positive_gt == False:
                TN_Pat_R_5 += 1
            elif patient_positive_5 != patient_positive_gt and patient_positive_gt == False:
                FP_Pat_R_5 += 1
            elif patient_positive_5 != patient_positive_gt and patient_positive_gt == True:
                FN_Pat_R_5 += 1          









    print(num_examples, num_tumor)
    print(P,N)
    print(TP, TN, FP, FN)
    print(TP_R_3, TN_R_3, FP_R_3, FN_R_3,'\n',TP_R_4, TN_R_4, FP_R_4, FN_R_4,'\n',TP_R_5, TN_R_5, FP_R_5, FN_R_5)
    print('---patient level-------')
    print(P_Pat, N_Pat)
    print(TP_Pat, TN_Pat, FP_Pat, FN_Pat)
    print(TP_Pat_R_3, TN_Pat_R_3, FP_Pat_R_3, FN_Pat_R_3)
    print(TP_Pat_R_4, TN_Pat_R_4, FP_Pat_R_4, FN_Pat_R_4)
    print(TP_Pat_R_5, TN_Pat_R_5, FP_Pat_R_5, FN_Pat_R_5)
    return TP/P, TN/N





 
                    
@torch.no_grad()
def inference_net_ucla_ai(
        model,
        inference_loader,
        device='cpu',
        pirads_sheet = None,
        pirads_2check=[3, 4],
        norad=False
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.eval()
    num_examples = 0
    num_tumor = 0
    P, TP, N, TN, FP, FN = 0, 0, 0, 0, 0, 0

    
    
    P_Pat, N_Pat = 0, 0
    TP_Pat, TN_Pat, FP_Pat, FN_Pat = 0, 0, 0, 0
    

    
    priads_sheet = pd.read_excel(pirads_sheet)
    for batch in inference_loader:
        img, label = batch["img"].to(device), batch["label"].to(device)

        
        if not norad:
            tumor = (label != 0).float()
            img = torch.cat([
                img,
                tumor
            ], dim=1)
        output = model(img)
        output = inference_post_process(output)
        
        
        
        position_list, num_unique = count_unique_contours(label)
        patient_name, patient_uid = batch['patient_name'][0], batch['patient_uid'][0]


        filtered_df = priads_sheet[priads_sheet['Patient ID'] == patient_name]
        filtered_df = filtered_df[filtered_df['seriesInstanceUID_MR'] == patient_uid]
        filtered_df = filtered_df.drop_duplicates(subset='ROI Volume (cc)')
        
        
        volumn_list = filtered_df['ROI Volume (cc)'].tolist()
        pirads_list = filtered_df['UCLA Score (Similar to PIRADS v2)'].tolist()
        # Combine and sort by list_a
        combined = sorted(zip(volumn_list, pirads_list), reverse=True)  # ascending sort
        # Unzip back
        volumn_list, pirads_list = zip(*combined)
        position_list = sorted(position_list, key=lambda x: x.shape[0], reverse=True)
        
        # print(volumn_list, pirads_list)
        # print([x.shape[0] for x in position_list])
        
        
        if len(filtered_df) == num_unique:
            num_examples+=1
            num_tumor += num_unique
            
            patient_positive_pred=False
            patient_positive_gt=False
            for pirads, index in zip(pirads_list, position_list):
                # print(index.shape, output.shape)
                # num_voxcel = index.shape[0]
                pathlogy_gt = label[0, 0, index[:, 2], index[:, 3], index[:, 4]]
                pathlogy_gt = torch.unique(pathlogy_gt).item()
                if pathlogy_gt == 1:
                    P += 1
                else:
                    N += 1
                prediction = output[0, index[:, 2], index[:, 3], index[:, 4]]
                count_1 = (prediction == 1).sum().item()  # Count number of 1's
                count_2 = (prediction == 2).sum().item()  # Count number of 2's
                
                
                if pirads < min(pirads_2check):
                    lesion_positive_prediction = False
                elif pirads > max(pirads_2check): 
                    lesion_positive_prediction = True
                else: # all pirads 2 check lesions
                    if count_1 > count_2:
                        lesion_positive_prediction = True
                    else:
                        lesion_positive_prediction = False
                        
                if lesion_positive_prediction == True and pathlogy_gt == 1:
                    TP += 1
                elif lesion_positive_prediction == True and pathlogy_gt == 2:
                    FP += 1
                elif lesion_positive_prediction == False and pathlogy_gt == 2:
                    TN += 1
                elif lesion_positive_prediction == False and pathlogy_gt == 1:
                    FN += 1
                    
                    
                if lesion_positive_prediction:
                    patient_positive_pred = True

                if pathlogy_gt == 1:
                    patient_positive_gt = True
            
            if patient_positive_pred == True and patient_positive_gt == True:
                TP_Pat += 1
            elif patient_positive_pred == True and patient_positive_gt == False:
                FP_Pat += 1
            elif patient_positive_pred == False and patient_positive_gt == False:
                TN_Pat += 1
            elif patient_positive_pred == False and patient_positive_gt == True:
                FN_Pat += 1









    print(num_examples, num_tumor)
    print(P,N)
    print(TP, TN, FP, FN)
    print('---patient level-------')
    print(P_Pat, N_Pat)
    print(TP_Pat, TN_Pat, FP_Pat, FN_Pat)
    return TP/P, TN/N









def get_rad_each_ucla(
        inference_loader,
        pirads_sheet = None,
        ucla_2check=3,
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch


    
    P, N = 0, 0
    

    
    priads_sheet = pd.read_excel(pirads_sheet)
    for batch in inference_loader:
        label = batch["label"]

        
        
        
        
        position_list, num_unique = count_unique_contours(label)
        patient_name, patient_uid = batch['patient_name'][0], batch['patient_uid'][0]


        filtered_df = priads_sheet[priads_sheet['Patient ID'] == patient_name]
        filtered_df = filtered_df[filtered_df['seriesInstanceUID_MR'] == patient_uid]
        filtered_df = filtered_df.drop_duplicates(subset='ROI Volume (cc)')
        
        
        volumn_list = filtered_df['ROI Volume (cc)'].tolist()
        pirads_list = filtered_df['UCLA Score (Similar to PIRADS v2)'].tolist()
        # Combine and sort by list_a
        combined = sorted(zip(volumn_list, pirads_list), reverse=True)  # ascending sort
        # Unzip back
        volumn_list, pirads_list = zip(*combined)
        position_list = sorted(position_list, key=lambda x: x.shape[0], reverse=True)
        
        # print(volumn_list, pirads_list)
        # print([x.shape[0] for x in position_list])
        
        
        if len(filtered_df) == num_unique:
            for pirads, index in zip(pirads_list, position_list):
                if pirads == ucla_2check:
                    # print(index.shape, output.shape)
                    # num_voxcel = index.shape[0]
                    pathlogy_gt = label[0, 0, index[:, 2], index[:, 3], index[:, 4]]
                    pathlogy_gt = torch.unique(pathlogy_gt).item()
                    if pathlogy_gt == 1:
                        P += 1
                    else:
                        N += 1
    print(P,N)






@torch.no_grad()
def inference_net_lesion_ucla_only(
        model,
        inference_loader,
        device='cpu',
        pirads_sheet = None,
        ucla_2check = 3
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.eval()
    num_examples = 0
    num_tumor = 0
    P, TP, N, TN, FP, FN = 0, 0, 0, 0, 0, 0


    
    
    priads_sheet = pd.read_excel(pirads_sheet)
    for batch in inference_loader:
        img, label = batch["img"].to(device), batch["label"].to(device)



        tumor = (label != 0).float()
        img = torch.cat([
            img,
            tumor
        ], dim=1)
        
        
        
        output = model(img)
        output = inference_post_process(output)
        
        
        
        position_list, num_unique = count_unique_contours(label)
        patient_name, patient_uid = batch['patient_name'][0], batch['patient_uid'][0]


        filtered_df = priads_sheet[priads_sheet['Patient ID'] == patient_name]
        filtered_df = filtered_df[filtered_df['seriesInstanceUID_MR'] == patient_uid]
        filtered_df = filtered_df.drop_duplicates(subset='ROI Volume (cc)')
        
        
        volumn_list = filtered_df['ROI Volume (cc)'].tolist()
        pirads_list = filtered_df['UCLA Score (Similar to PIRADS v2)'].tolist()
        # Combine and sort by list_a
        combined = sorted(zip(volumn_list, pirads_list), reverse=True)  # ascending sort
        # Unzip back
        volumn_list, pirads_list = zip(*combined)
        position_list = sorted(position_list, key=lambda x: x.shape[0], reverse=True)
        
        # print(volumn_list, pirads_list)
        # print([x.shape[0] for x in position_list])
        
        
        if len(filtered_df) == num_unique:
            num_examples+=1
            num_tumor += num_unique
            
            patient_positive_ai,patient_positive_3,patient_positive_4,patient_positive_5=False,False,False,False
            for pirads, index in zip(pirads_list, position_list):
                if pirads == ucla_2check:
                    pathlogy_gt = label[0, 0, index[:, 2], index[:, 3], index[:, 4]]
                    pathlogy_gt = torch.unique(pathlogy_gt).item()
                    if pathlogy_gt == 1:
                        P += 1
                    else:
                        N += 1
                    prediction = output[0, index[:, 2], index[:, 3], index[:, 4]]
                    count_1 = (prediction == 1).sum().item()  # Count number of 1's
                    count_2 = (prediction == 2).sum().item()  # Count number of 2's
                    
                    if count_1 > count_2 and pathlogy_gt == 1:
                        TP += 1
                    elif count_1 < count_2 and pathlogy_gt == 2:
                        TN += 1
                    if count_1 > count_2 and pathlogy_gt == 2:
                        FP += 1
                    elif count_1 < count_2 and pathlogy_gt == 1:
                        FN += 1
                    
    print(P, N)
    print(TP, FP, FN, TN)
    
    
    
    
    
    
    
    
def inference_heatmap_post_process(o, pirads, index):
    pirads /= 5
    
    
    
    o = torch.softmax(o, dim=1)[:, 1, ...].unsqueeze(1)
    
    pirads_volumn = torch.zeros_like(o)
    pirads_volumn[0, 0, index[:, 2], index[:, 3], index[:, 4]] = pirads
    
    
    o += pirads_volumn
    o /= 2
    
    
    return o 
@torch.no_grad()
def inference_net_heatmap(
        model,
        inference_loader,
        device='cpu',
        pirads_sheet = None,
        no_rad=False
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.eval()
    num_examples = 0
    num_tumor = 0
    P, TP, N, TN, FP, FN = 0, 0, 0, 0, 0, 0

    
    
    P_Pat, N_Pat = 0, 0
    TP_Pat, TN_Pat, FP_Pat, FN_Pat = 0, 0, 0, 0
    

    
    
    priads_sheet = pd.read_excel(pirads_sheet)
    for batch in inference_loader:
        img, label = batch["img"].to(device), batch["label"].to(device)



        if not no_rad:
            tumor = (label != 0).float()
            img = torch.cat([
                img,
                tumor
            ], dim=1)
        
        output = model(img)

        
        
        
        position_list, num_unique = count_unique_contours(label)
        patient_name, patient_uid = batch['patient_name'][0], batch['patient_uid'][0]


        filtered_df = priads_sheet[priads_sheet['Patient ID'] == patient_name]
        filtered_df = filtered_df[filtered_df['seriesInstanceUID_MR'] == patient_uid]
        filtered_df = filtered_df.drop_duplicates(subset='ROI Volume (cc)')
        
        
        volumn_list = filtered_df['ROI Volume (cc)'].tolist()
        pirads_list = filtered_df['UCLA Score (Similar to PIRADS v2)'].tolist()
        # Combine and sort by list_a
        combined = sorted(zip(volumn_list, pirads_list), reverse=True)  # ascending sort
        # Unzip back
        volumn_list, pirads_list = zip(*combined)
        position_list = sorted(position_list, key=lambda x: x.shape[0], reverse=True)
        

        
        
        if len(filtered_df) == num_unique:
            num_examples+=1
            num_tumor += num_unique
            
            patient_positive_ai = False
            for pirads, index in zip(pirads_list, position_list):
                o = inference_heatmap_post_process(output, pirads, index)
                pathlogy_gt = label[0, 0, index[:, 2], index[:, 3], index[:, 4]]
                pathlogy_gt = torch.unique(pathlogy_gt).item()
                if pathlogy_gt == 1:
                    P += 1
                else:
                    N += 1
                prediction = o[0, 0, index[:, 2], index[:, 3], index[:, 4]]
                count_1 = (prediction > 0.5).sum().item()  # Count number of 1's
                count_2 = (prediction <= 0.5).sum().item()  # Count number of 2's
                
                if count_1 > count_2 and pathlogy_gt == 1:
                    TP += 1
                elif count_1 < count_2 and pathlogy_gt == 2:
                    TN += 1
                if count_1 > count_2 and pathlogy_gt == 2:
                    FP += 1
                elif count_1 < count_2 and pathlogy_gt == 1:
                    FN += 1

                    
                    
                patient_positive_gt, patient_positive_ai = False, False
                if pathlogy_gt == 1:
                    patient_positive_gt=True
                if count_1>count_2:
                    patient_positive_ai=True

            
            if patient_positive_gt == True:
                P_Pat +=1
            else:
                N_Pat += 1
            
            
            if patient_positive_ai == patient_positive_gt and patient_positive_gt == True:
                TP_Pat += 1
            elif patient_positive_ai == patient_positive_gt and patient_positive_gt == False:
                TN_Pat += 1
            elif patient_positive_ai != patient_positive_gt and patient_positive_gt == False:
                FP_Pat += 1
            elif patient_positive_ai != patient_positive_gt and patient_positive_gt == True:
                FN_Pat += 1
    
                

    print(P, N)
    print(TP, FN, TN, FP)
    
    

    
    
    print(P_Pat, N_Pat)
    print(TP_Pat, FN_Pat, TN_Pat, FP_Pat)
    
    
    



@torch.no_grad()
def inference_net_ucla_ai_heatmap(
        model,
        inference_loader,
        device='cpu',
        pirads_sheet = None,
        pirads_2check=[3, 4],
        norad=False
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    model.to(device)
    model.eval()
    num_examples = 0
    num_tumor = 0
    P, TP, N, TN, FP, FN = 0, 0, 0, 0, 0, 0

    
    
    P_Pat, N_Pat = 0, 0
    TP_Pat, TN_Pat, FP_Pat, FN_Pat = 0, 0, 0, 0
    

    
    priads_sheet = pd.read_excel(pirads_sheet)
    for batch in inference_loader:
        img, label = batch["img"].to(device), batch["label"].to(device)

        
        if not norad:
            tumor = (label != 0).float()
            img = torch.cat([
                img,
                tumor
            ], dim=1)
        output = model(img)

        
        
        
        position_list, num_unique = count_unique_contours(label)
        patient_name, patient_uid = batch['patient_name'][0], batch['patient_uid'][0]


        filtered_df = priads_sheet[priads_sheet['Patient ID'] == patient_name]
        filtered_df = filtered_df[filtered_df['seriesInstanceUID_MR'] == patient_uid]
        filtered_df = filtered_df.drop_duplicates(subset='ROI Volume (cc)')
        
        
        volumn_list = filtered_df['ROI Volume (cc)'].tolist()
        pirads_list = filtered_df['UCLA Score (Similar to PIRADS v2)'].tolist()
        # Combine and sort by list_a
        combined = sorted(zip(volumn_list, pirads_list), reverse=True)  # ascending sort
        # Unzip back
        volumn_list, pirads_list = zip(*combined)
        position_list = sorted(position_list, key=lambda x: x.shape[0], reverse=True)
        
        # print(volumn_list, pirads_list)
        # print([x.shape[0] for x in position_list])
        
        
        if len(filtered_df) == num_unique:
            num_examples+=1
            num_tumor += num_unique
            
            patient_positive_pred=False
            patient_positive_gt=False
            for pirads, index in zip(pirads_list, position_list):
                
                o = inference_heatmap_post_process(output, pirads, index)
                
                pathlogy_gt = label[0, 0, index[:, 2], index[:, 3], index[:, 4]]
                pathlogy_gt = torch.unique(pathlogy_gt).item()
                if pathlogy_gt == 1:
                    P += 1
                else:
                    N += 1
                prediction = o[0, 0, index[:, 2], index[:, 3], index[:, 4]]
                count_1 = (prediction > 0.5).sum().item()  # Count number of 1's
                count_2 = (prediction <= 0.5).sum().item()  # Count number of 2's
                
                
                if pirads < min(pirads_2check):
                    lesion_positive_prediction = False
                elif pirads > max(pirads_2check): 
                    lesion_positive_prediction = True
                else: # all pirads 2 check lesions
                    if count_1 > count_2:
                        lesion_positive_prediction = True
                    else:
                        lesion_positive_prediction = False
                        
                if lesion_positive_prediction == True and pathlogy_gt == 1:
                    TP += 1
                elif lesion_positive_prediction == True and pathlogy_gt == 2:
                    FP += 1
                elif lesion_positive_prediction == False and pathlogy_gt == 2:
                    TN += 1
                elif lesion_positive_prediction == False and pathlogy_gt == 1:
                    FN += 1
                    
                    
                if lesion_positive_prediction:
                    patient_positive_pred = True

                if pathlogy_gt == 1:
                    patient_positive_gt = True
            
            if patient_positive_pred == True and patient_positive_gt == True:
                TP_Pat += 1
            elif patient_positive_pred == True and patient_positive_gt == False:
                FP_Pat += 1
            elif patient_positive_pred == False and patient_positive_gt == False:
                TN_Pat += 1
            elif patient_positive_pred == False and patient_positive_gt == True:
                FN_Pat += 1









    print(num_examples, num_tumor)
    print(P,N)
    print(TP, FN, TN, FP)
    print('---patient level-------')
    print(P_Pat, N_Pat)
    print(TP_Pat, FN_Pat, TN_Pat, FP_Pat)
    return TP/P, TN/N


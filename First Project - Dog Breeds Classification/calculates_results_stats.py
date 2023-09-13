#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/calculates_results_stats.py
#                                                                             
# PROGRAMMER: Ammar Taha Mohamedy
# DATE CREATED: 7/5/2023                                 
# REVISED DATE: 
# PURPOSE: Create a function calculates_results_stats that calculates the 
#          statistics of the results of the programrun using the classifier's model 
#          architecture to classify the images. This function will use the 
#          results in the results dictionary to calculate these statistics. 
#          This function will then put the results statistics in a dictionary
#          (results_stats_dic) that's created and returned by this function.
#          This will allow the user of the program to determine the 'best' 
#          model for classifying the images. The statistics that are calculated
#          will be counts and percentages. Please see "Intro to Python - Project
#          classifying Images - xx Calculating Results" for details on the 
#          how to calculate the counts and percentages for this function.    
#         This function inputs:
#            -The results dictionary as results_dic within calculates_results_stats 
#             function and results for the function call within main.
#         This function creates and returns the Results Statistics Dictionary -
#          results_stats_dic. This dictionary contains the results statistics 
#          (either a percentage or a count) where the key is the statistic's 
#           name (starting with 'pct' for percentage or 'n' for count) and value 
#          is the statistic's value.  This dictionary should contain the 
#          following keys:
#            n_images - number of images
#            n_dogs_img - number of dog images
#            n_notdogs_img - number of NON-dog images
#            n_match - number of matches between pet & classifier labels
#            n_correct_dogs - number of correctly classified dog images
#            n_correct_notdogs - number of correctly classified NON-dog images
#            n_correct_breed - number of correctly classified dog breeds
#            pct_match - percentage of correct matches
#            pct_correct_dogs - percentage of correctly classified dogs
#            pct_correct_breed - percentage of correctly classified dog breeds
#            pct_correct_notdogs - percentage of correctly classified NON-dogs
#
##
# TODO 5: Define calculates_results_stats function below, please be certain to replace None
#       in the return statement with the results_stats_dic dictionary that you create 
#       with this function
# 
def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the program run using classifier's model 
    architecture to classifying pet images. Then puts the results statistics in a 
    dictionary (results_stats_dic) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that 
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
    Returns:
     results_stats_dic - Dictionary that contains the results statistics (either
                    a percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value. See comments above
                     and the previous topic Calculating Results in the class for details
                     on how to calculate the counts and statistics.
    """        
    # Replace None with the results_stats_dic dictionary that you created with this function 
    results_stats_dic = {"n_images": None,          "n_dogs_img":     None, "n_notdogs_img":     None, 
                         "n_match":  None,          "n_correct_dogs": None, "n_correct_notdogs": None, 
                         "n_correct_breed":   None, "pct_match":      None, "pct_correct_dogs":  None,
                         "pct_correct_breed": None, "pct_correct_notdogs": None}
    
    """" 
    Calculating Counts
    Here I use a list comprehension to build a list of True(1) when a condition is met, and assign the 
    length of result list to be the value of the corresponding count key in the stats dict.
    """
    # 1. Z: Number of Images("n_images"): length of results_dic
    Z = len(results_dic)
    results_stats_dic["n_images"] = Z
    # 2. B: Number of Dog Images("n_dogs_img"): Pet Label is a dog: values[3] = 1
    B = len([1 for values in results_dic.values() if values[3] == 1])
    results_stats_dic["n_dogs_img"] = B
    # 3. D: Number of Not Dog Images("n_notdogs_img"): Pet Label is NOT a dog: values[3] = 0
    D = len([1 for values in results_dic.values() if values[3] == 0])
    results_stats_dic["n_notdogs_img"] = D
    # 4. Y: Number of matches between pet & classifier labels("n_match"): values[2] =1
    Y = len([1 for values in results_dic.values() if values[2] == 1])
    results_stats_dic["n_match"] = Y
    # 5. A: Number of Correct Dog("n_correct_dogs"): Both labels are of dogs: values[3] = 1 and values[4] = 1
    A = len([1 for values in results_dic.values() if values[3] == 1 and values[4] == 1])
    results_stats_dic["n_correct_dogs"] = A
    # 6. C: Num of Correct Non-Dog("n_correct_notdogs"): Both labels NOT dogs: values[3] = 0 and values[4] = 0
    C = len([1 for values in results_dic.values() if values[3] == 0 and values[4] == 0])
    results_stats_dic["n_correct_notdogs"] = C
    # 7. E: Num of Correct Breed("n_correct_breed"): Pet Label is dog & Labels match: values[3] = 1 and values[2] = 1
    E = len([1 for values in results_dic.values() if values[3] == 1 and values[2] == 1])
    results_stats_dic["n_correct_breed"] = E
    
    ### Calculating Percentages
    """
    When calculating the percentage of correctly classified Non-Dog Images, 
    use a conditional statement to check that D, the number of "not-a-dog" images, is greater than zero. 
    To avoid division by zero error, only if D is greater than zero should C/D be computed; otherwise, 
    this should be set to 0.
    """
    # 1. Percentage Label Matches("pct_match"): Y/Z * 100 
    results_stats_dic["pct_match"] = (Y / Z) * 100
    # 2. Percentage of Correctly Classified Dog Images("pct_correct_dogs"): A/B * 100
    results_stats_dic["pct_correct_dogs"] = (A / B) * 100
    # 3. Percentage of Correctly Classified Dog Breeds("pct_correct_breed"): E/B * 100
    results_stats_dic["pct_correct_breed"] = (E / B) * 100
    # 4. Percentage of Correctly Classified Non-Dog Images("pct_correct_notdogs"): C/D * 100 if D > 0
    if D > 0:
        pct_correct_notdogs = (C / D) * 100
    else:
        pct_correct_notdogs = 0
    results_stats_dic["pct_correct_notdogs"] = pct_correct_notdogs

    return results_stats_dic
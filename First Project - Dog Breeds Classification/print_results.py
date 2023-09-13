#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results.py
#                                                                             
# PROGRAMMER: Ammar Taha Mohamedy
# DATE CREATED: 7/5/2023
# REVISED DATE: 
# PURPOSE: Create a function print_results that prints the results statistics
#          from the results statistics dictionary (results_stats_dic). It 
#          should also allow the user to be able to print out cases of misclassified
#          dogs and cases of misclassified breeds of dog using the Results 
#          dictionary (results_dic).  
#         This function inputs:
#            -The results dictionary as results_dic within print_results 
#             function and results for the function call within main.
#            -The results statistics dictionary as results_stats_dic within 
#             print_results function and results_stats for the function call within main.
#            -The CNN model architecture as model wihtin print_results function
#             and in_arg.arch for the function call within main. 
#            -Prints Incorrectly Classified Dogs as print_incorrect_dogs within
#             print_results function and set as either boolean value True or 
#             False in the function call within main (defaults to False)
#            -Prints Incorrectly Classified Breeds as print_incorrect_breed within
#             print_results function and set as either boolean value True or 
#             False in the function call within main (defaults to False)
#         This function does not output anything other than printing a summary
#         of the final results.
##
# TODO 6: Define print_results function below, specifically replace the None
#       below by the function definition of the print_results function. 
#       Notice that this function doesn't to return anything because it  
#       prints a summary of the results using results_dic and results_stats_dic
# 
def print_results(results_dic, results_stats_dic, model, 
                  print_incorrect_dogs = False, print_incorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
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
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - Indicates which CNN model architecture will be used by the 
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    # A statement that indicates which of the three CNN model architectures is being used in this run.
    print("This Run of The Program uses The CNN Model: {}\n".format(model))

    ### Printing Counts from the results_stats_dic
    # Number of Images
    print("The Total Number of Images in the Project is: {}\n".format(results_stats_dic["n_images"]))
    # Number of Dog Images
    print("The Number of Dog Images is: {}\n".format(results_stats_dic["n_dogs_img"]))
    # Number of "Not-a" Dog Images
    print("The Number of Images Labeled as Not-a-Dog is: {}\n".format(results_stats_dic["n_notdogs_img"]))
    # The Number of dog breed images
    print("The Number of correctly classified dog breeds: {}\n".format(results_stats_dic["n_correct_breed"]))
    ### Printing the percentages - keys of the results_stats_dic that starts with 'p'
    # NOTE that the description follows the order of the percentage keys in the dictionary from calculate_results_stats.py
    perc_desc = ['Percentage Label Matches', 'Percentage of Correctly Classified Dog Images', 
                 'Percentage of Correctly Classified Dog Breeds', 
                 'Percentage of Correctly Classified Non-Dog Images']
    counter = 0
    for key in results_stats_dic.keys():
        if key[0] == 'p':
            print("The Stats of {} 'The {}' is {}\n".format(key, perc_desc[counter], results_stats_dic[key]))
            counter += 1
    
    ### Printing Misclassified Dogs
    if (print_incorrect_dogs and 
        ( (results_stats_dic["n_correct_dogs"] + results_stats_dic["n_correct_notdogs"] )
         != results_stats_dic["n_images"] )
       ):
        print("Misclassified Dog Labels: \n")
        for key in results_dic.keys():
            if (results_dic[key][3] == 1 and results_dic[key][4] == 0) or (results_dic[key][3] == 0 and results_dic[key][4] == 1): 
                print("This Pet Entry that has Pet Image Label of {} and Classifier Label of {} is Misclassified by The Model\n".format(results_dic[key][0],results_dic[key][1]))

    ### Printing Misclassified Dog Breeds
    if print_incorrect_breed and (results_stats_dic["n_correct_dogs"] != results_stats_dic["n_correct_breed"]):
        print("Misclassified Dog Breeds: \n")
        for key in results_dic.keys():
            if (sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0):
                print("This Pet Entry that has Pet Image Label of {} and Classifier Label of {} is Breed-Misclassified by The Model\n".format(results_dic[key][0],results_dic[key][1]))
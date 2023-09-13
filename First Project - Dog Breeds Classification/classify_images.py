#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classify_images.py
#                                                                             
# PROGRAMMER: Ammar Taha Mohamedy
# DATE CREATED: 7/4/2023                                
# REVISED DATE: 
# PURPOSE: Create a function classify_images that uses the classifier function 
#          to create the classifier labels and then compares the classifier 
#          labels to the pet image labels. This function inputs:
#            -The Image Folder as image_dir within classify_images and function 
#             and as in_arg.dir for function call within main. 
#            -The results dictionary as results_dic within classify_images 
#             function and results for the functin call within main.
#            -The CNN model architecture as model wihtin classify_images function
#             and in_arg.arch for the function call within main. 
#           This function uses the extend function to add items to the list 
#           that's the 'value' of the results dictionary. You will be adding the
#           classifier label as the item at index 1 of the list and the comparison 
#           of the pet and classifier labels as the item at index 2 of the list.
#
##
# Imports classifier function for using CNN to classify images 
from classifier import classifier 

# TODO 3: Define classify_images function below, specifically replace the None
#       below by the function definition of the classify_images function. 
#       Notice that this function doesn't return anything because the 
#       results_dic dictionary that is passed into the function is a mutable 
#       data type so no return is needed.
# 
def classify_images(images_dir, results_dic, model):
    """
    Creates classifier labels with classifier function, compares pet labels to 
    the classifier labels, and adds the classifier label and the comparison of 
    the labels to the results dictionary using the extend function. Be sure to
    format the classifier labels so that they will match your pet image labels.
    The format will include putting the classifier labels in all lower case 
    letters and strip the leading and trailing whitespace characters from them.
    For example, the Classifier function returns = 'Maltese dog, Maltese terrier, Maltese' 
    so the classifier label = 'maltese dog, maltese terrier, maltese'.
    Recall that dog names from the classifier function can be a string of dog 
    names separated by commas when a particular breed of dog has multiple dog 
    names associated with that breed. For example, you will find pet images of
    a 'dalmatian'(pet label) and it will match to the classifier label 
    'dalmatian, coach dog, carriage dog' if the classifier function correctly 
    classified the pet images of dalmatians.
     PLEASE NOTE: This function uses the classifier() function defined in 
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the 
     classifier() function to classify images within this function 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be
                   classified by the classifier function (string)
      results_dic - Results Dictionary with 'key' as image filename and 'value'
                    as a List. Where the list will contain the following items: 
                  index 0 = pet image label (string)
                --- where index 1 & index 2 are added by this function ---
                  NEW - index 1 = classifier label (string)
                  NEW - index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifer labels and 0 = no match between labels
      model - Indicates which CNN model architecture will be used by the 
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
     Returns:
           None - results_dic is mutable data type so no return needed.         
    """
    ## Using the classifier function to obtain labels for pet images using certain model, this is done by
    # looping and attaching the images_dir (which defaults to "pet_images/") and the filename for the pet images
    # that are the keys of the results_dic dictionary returned by get_pet_labels() when calling this function in main().      
    classifier_labels = [] # index 1 of the results_dic to be updated by this function.
    for i in range(len(results_dic)): # This assumes that images_dir is given with a "/"
      classified_label = classifier("{}{}".format(images_dir,list(results_dic.keys())[i]),model).lower()
      classifier_labels.append(classified_label)
    
    ## The Comparison - index 2 of the results_dic
    # Extracting the Pet Images Label - NOTE that a comprehension is used to return a list of strings
    #   to be comparable with the list of strings of the classifier function. So, list(results_dic.values())
    #   isn't correct becuase that returns a list of lists.
    truth_labels = [value[0] for value in results_dic.values()]
    # Initializing a list of zeros for the comparison result
    comp_list = [0 for _ in range(len(truth_labels))]
    # Defining a function that compares two lists built suitable for our case
    def label_compare(a, b): # a: truth_labels, b: classifier labels
      for i in range(len(a)):
        m = b[i].split(',')
        for j in range(len(m)):
          if a[i] == m[j].strip():
            comp_list[i] = 1 
    # Using the function to tweak the comparison 1/0 list
    label_compare(truth_labels, classifier_labels) # This line adjusts the comp_list

    ## Extending the results_dic
    # This is done by building and using a function that extends the list-based values() of a dictionary by an input list
    # INPUTS: base_dict is results_dic, and list_to_extend is classifier_labels and/or comp_list
    def extend_dict(base_dict, list_to_extend): 
      for i in range(len(list_to_extend)):
        # NOTE: The input to the extend() method is made to be list to account for object that's not list to be inserted i.e. appending an int.
        base_dict[list(base_dict.keys())[i]].extend([list_to_extend[i]])
    # Using the function twice, first to add the classifier labels, and then for the comparison ints
    extend_dict(results_dic, classifier_labels)
    extend_dict(results_dic, comp_list)
# TOOL USED TO AUTOMATICALLY MOVE THE DATASET FILES FROM AUGMENTATON DIRECTORY TO THE CORRESPONDING DATASET SUBFOLDERS
import os

folderStructurePath = '../datasets/image-corruption-dataset/cancer-noise-015'

# Make core folders and their sub-folders
os.makedirs(folderStructurePath + '/' + 'all-corrupted')
os.makedirs(folderStructurePath + '/all-corrupted/train')
os.makedirs(folderStructurePath + '/all-corrupted/train/benign')
os.makedirs(folderStructurePath + '/all-corrupted/train/malignant')
os.makedirs(folderStructurePath + '/all-corrupted/validation')
os.makedirs(folderStructurePath + '/all-corrupted/validation/benign')
os.makedirs(folderStructurePath + '/all-corrupted/validation/malignant')

os.makedirs(folderStructurePath + '/' + 'control')
os.makedirs(folderStructurePath + '/control/train')
os.makedirs(folderStructurePath + '/control/train/benign')
os.makedirs(folderStructurePath + '/control/train/malignant')
os.makedirs(folderStructurePath + '/control/validation')
os.makedirs(folderStructurePath + '/control/validation/benign')
os.makedirs(folderStructurePath + '/control/validation/malignant')

os.makedirs(folderStructurePath + '/' + 'training-corrupted')
os.makedirs(folderStructurePath + '/training-corrupted/train')
os.makedirs(folderStructurePath + '/training-corrupted/train/benign')
os.makedirs(folderStructurePath + '/training-corrupted/train/malignant')
os.makedirs(folderStructurePath + '/training-corrupted/validation')
os.makedirs(folderStructurePath + '/training-corrupted/validation/benign')
os.makedirs(folderStructurePath + '/training-corrupted/validation/malignant')

os.makedirs(folderStructurePath + '/' + 'validation-corrupted')
os.makedirs(folderStructurePath + '/validation-corrupted/train')
os.makedirs(folderStructurePath + '/validation-corrupted/train/benign')
os.makedirs(folderStructurePath + '/validation-corrupted/train/malignant')
os.makedirs(folderStructurePath + '/validation-corrupted/validation')
os.makedirs(folderStructurePath + '/validation-corrupted/validation/benign')
os.makedirs(folderStructurePath + '/validation-corrupted/validation/malignant')

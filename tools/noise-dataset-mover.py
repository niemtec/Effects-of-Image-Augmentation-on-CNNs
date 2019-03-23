# TOOL USED TO AUTOMATICALLY MOVE THE DATASET FILES FROM AUGMENTATON DIRECTORY TO THE CORRESPONDING DATASET SUBFOLDERS
import os

folderStructurePath = '../datasets/image-corruption-dataset/cats-dogs-noise-004'

# Make core folders and their sub-folders
os.makedirs(folderStructurePath + '/' + 'all-corrupted')
os.makedirs(folderStructurePath + '/all-corrupted/train')
os.makedirs(folderStructurePath + '/all-corrupted/train/cat')
os.makedirs(folderStructurePath + '/all-corrupted/train/dog')
os.makedirs(folderStructurePath + '/all-corrupted/validation')
os.makedirs(folderStructurePath + '/all-corrupted/validation/cat')
os.makedirs(folderStructurePath + '/all-corrupted/validation/dog')

os.makedirs(folderStructurePath + '/' + 'control')
os.makedirs(folderStructurePath + '/control/train')
os.makedirs(folderStructurePath + '/control/train/cat')
os.makedirs(folderStructurePath + '/control/train/dog')
os.makedirs(folderStructurePath + '/control/validation')
os.makedirs(folderStructurePath + '/control/validation/cat')
os.makedirs(folderStructurePath + '/control/validation/dog')

os.makedirs(folderStructurePath + '/' + 'training-corrupted')
os.makedirs(folderStructurePath + '/training-corrupted/train')
os.makedirs(folderStructurePath + '/training-corrupted/train/cat')
os.makedirs(folderStructurePath + '/training-corrupted/train/dog')
os.makedirs(folderStructurePath + '/training-corrupted/validation')
os.makedirs(folderStructurePath + '/training-corrupted/validation/cat')
os.makedirs(folderStructurePath + '/training-corrupted/validation/dog')

os.makedirs(folderStructurePath + '/' + 'validation-corrupted')
os.makedirs(folderStructurePath + '/validation-corrupted/train')
os.makedirs(folderStructurePath + '/validation-corrupted/train/cat')
os.makedirs(folderStructurePath + '/validation-corrupted/train/dog')
os.makedirs(folderStructurePath + '/validation-corrupted/validation')
os.makedirs(folderStructurePath + '/validation-corrupted/validation/cat')
os.makedirs(folderStructurePath + '/validation-corrupted/validation/dog')

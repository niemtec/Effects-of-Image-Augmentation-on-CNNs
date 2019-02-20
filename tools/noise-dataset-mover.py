# TOOL USED TO AUTOMATICALLY MOVE THE DATASET FILES FROM AUGMENTATON DIRECTORY TO THE CORRESPONDING DATASET SUBFOLDERS

folderStructurePath = '/datasets/image-corruption-dataset/cats-dogs-noise-003'

# Make core folders and their sub-folders
os.mkdir(folderStructurePath + '/' + 'all-corrupted')
os.mkdir(folderStructurePath + '/all-corrupted/train')
os.mkdir(folderStructurePath + '/all-corrupted/train/cat')
os.mkdir(folderStructurePath + '/all-corrupted/train/dog')
os.mkdir(folderStructurePath + '/all-corrupted/validation')
os.mkdir(folderStructurePath + '/all-corrupted/validation/cat')
os.mkdir(folderStructurePath + '/all-corrupted/validation/dog')

os.mkdir(folderStructurePath + '/' + 'control')
os.mkdir(folderStructurePath + '/control/train')
os.mkdir(folderStructurePath + '/control/train/cat')
os.mkdir(folderStructurePath + '/control/train/dog')
os.mkdir(folderStructurePath + '/control/validation')
os.mkdir(folderStructurePath + '/control/validation/cat')
os.mkdir(folderStructurePath + '/control/validation/dog')

os.mkdir(folderStructurePath + '/' + 'training-corrupted')
os.mkdir(folderStructurePath + '/training-corrupted/train')
os.mkdir(folderStructurePath + '/training-corrupted/train/cat')
os.mkdir(folderStructurePath + '/training-corrupted/train/dog')
os.mkdir(folderStructurePath + '/training-corrupted/validation')
os.mkdir(folderStructurePath + '/training-corrupted/validation/cat')
os.mkdir(folderStructurePath + '/training-corrupted/validation/dog')

os.mkdir(folderStructurePath + '/' + 'validation-corrupted')
os.mkdir(folderStructurePath + '/validation-corrupted/train')
os.mkdir(folderStructurePath + '/validation-corrupted/train/cat')
os.mkdir(folderStructurePath + '/validation-corrupted/train/dog')
os.mkdir(folderStructurePath + '/validation-corrupted/validation')
os.mkdir(folderStructurePath + '/validation-corrupted/validation/cat')
os.mkdir(folderStructurePath + '/validation-corrupted/validation/dog')

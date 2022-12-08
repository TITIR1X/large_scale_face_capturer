import cv2, os

try:
 os.system('color 6')
 os.system('cls')
except:os.system('clear')
 
print("""
 ./capture_face_from_images.py
  ____              _______ _   ______ _       ___ __     __
 |  _ \            /__   __(_)/__   __(_) _ __|_| |\ \\\\  / /
 | |_) |_   _         | |   _    | |   _ | '__| | | \ \\\\/ /
 |  _ <| | | |        | | 0| |   | |  | || |    | |  \  \\\\
 | |_) | |_| |        | | /| |   | |  | || |    | | / /\ \\\\
 |____/ \__, |        |_| /|_|   |_|  |_||_|    |_|/_/  \_\\\\
         __/ |                                               
        |___/                           
""")


faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

input_folder_path = input('Folder with images to capture: ')
input_folder_path = input_folder_path.replace("\\", '/')
input_folder_path = input_folder_path.replace('"', '')

output_folder = input(f'Output folder name: ./')
if output_folder in '':
    output_folder = f'{input_folder_path}/Faces_captured_MF'
else:
    os.mkdir(output_folder)

print('Working... Progress will be displayed as each folder is completed.')

for folder in os.listdir(input_folder_path):
    img_count = 1
    os.mkdir(f'{output_folder}/{folder}')
    imagesList = os.listdir(f'{input_folder_path}/{folder}')

    for imageName in imagesList:
        image = cv2.imread(f'{input_folder_path}/{folder}/{imageName}')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray,1.3, 5)

        for (x,y,w,h) in faces:
            face = image[y:y+h,x:x+w]
            face = cv2.resize(face,(500, 500), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f'{output_folder}/{folder}/r{img_count}.jpg', face)
            img_count +=1

    print(f'\n\nfolder: {input_folder_path}/{folder} ok!\n\n')

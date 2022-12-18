from email.mime import image
from PIL import Image
import cv2, os, shutil, webbrowser, platform

# Limpiar la consola
os_name = platform.system()

if os_name == 'Windows':
    os.system('cls')
elif os_name == "Linux":
    os.system('clear')

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

# Solicitar la ruta de la carpeta con las imágenes a procesar
images_path = input('Folder with images to capture: ')
# Eliminar barras invertidas y comillas dobles
images_path = images_path.replace("\\", '/')
images_path = images_path.replace('"', '')

# Solicitar la ruta de la carpeta de salida
output_folder = input(f'Output folder name: ./')
# Si no se especifica, se crea una carpeta en la misma ubicación que la carpeta de imágenes
if output_folder in '':
    output_folder = f'{images_path}/Captured_faces'
    os.system(f'md {output_folder}')
# Si se especifica una carpeta de salida, se crea
else:
    os.system(f'md {output_folder}')
# Eliminar barras invertidas
output_folder = output_folder.replace("\\", '/')

# Obtener la lista de imágenes en la carpeta especificada
images_path_list = os.listdir(images_path)

# Cargar el clasificador de rostros
face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar contador
count = 0

# Recorrer la lista de imágenes
for image_name in images_path_list:

    # Leer la imagen
    image = cv2.imread(images_path + '/' + image_name)
    try:
        # Crear copia de la imagen
        image_aux = image.copy()
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        faces = face_classif.detectMultiScale(gray, 1.3, 10)

        # Recorrer las coordenadas de los rostros detectados
        for (x, y, w, h) in faces:
            # Extraer el rostro de la imagen
            face = image_aux[y:y + h, x:x + w]
            # Redimensionar el rostro
            face = cv2.resize(face, (500, 500), interpolation=cv2.INTER_CUBIC)
            # Guardar el rostro en la carpeta de salida
            cv2.imwrite(output_folder + '/r{}.jpeg'.format(count), face)
            # Mostrar mensaje de progreso
            print(f'[{count} of {len(images_path_list)}] -> r{count}.jpg ..ok')
            # Incrementar contador
            count += 1
    except:
        # En caso de error, continuar con la siguiente imagen
        pass
    
# Finalizar abriendo la carpeta de salida en el explorador de archivos.
webbrowser.open(os.path.realpath(output_folder))

print('\ncapture_face_from_images.py: Programa finalizado.')
exit()

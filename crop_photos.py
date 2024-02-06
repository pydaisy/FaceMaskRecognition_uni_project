import cv2
import os

# Ścieżka do katalogu z dwiema kategoriami: "mask" i "no_mask"
dataset_path = 'training_dataset'
categories = ['mask', 'no_mask']

if not os.path.exists('training_dataset_clean'):
    os.makedirs('training_dataset_clean')
    os.makedirs('training_dataset_clean/mask')
    os.makedirs('training_dataset_clean/no_mask')

# Detektora twarzy (przy użyciu domyślnego klasyfikatora Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Przeszukiwanie kategorii:
for category in categories:
    category_path = os.path.join(dataset_path, category)

    # Wczytywanie zdjęcia z katalogu po kolei
    for filename in os.listdir(category_path):
        if filename.lower().endswith(('.png')):
            # Pełna ścieżka do pliku zdjęcia
            image_path = os.path.join(category_path, filename)

            # Wczytanie zdjęciu
            image = cv2.imread(image_path)

            # Konwertowanie obrazu na odcienie szarości
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Wykrywanie twarzy na danym zdjęciu
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Przycinanie zdjęcie do obszaru twarzy (pierwsza znaleziona twarz)
            for (x, y, w, h) in faces:
                cropped_face = image[y:y+h, x:x+w]

                # Zapisz przyciętą twarz do nowego plik
                output_path = os.path.join(f'training_dataset_clean/{category}', f'{filename}')
                cv2.imwrite(output_path, cropped_face)

            # Wyświetlenie przyciętej twarzy
            if len(faces) > 0:
                cv2.imshow('Cropped Face', cropped_face)
                cv2.waitKey(1)  # Czekaj 1 ms, aby pozwolić na wyświetlenie okna, ale natychmiast je zamknij
                cv2.destroyAllWindows()

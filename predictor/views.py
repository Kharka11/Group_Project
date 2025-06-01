from django.shortcuts import render

# Create your views here.
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import torch
from torchvision import transforms
from PIL import Image
import logging

# Import your LeNet5 model definition
from .models import LeNet5

# Configure logging
logger = logging.getLogger(__name__)

# Path to the trained model file
model_path = os.path.join(settings.BASE_DIR, "lenet_final.pth")

# Load the model
model = LeNet5(num_classes=13)  # Ensure this matches your training configuration
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define transformations matching the training setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match the training image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define class names and descriptions
class_descriptions = {
    'Bird': ("Jaa", 
             "Often depicted as a rooster or generic bird, this mask represents vigilance, awakening, and the passage of time. The dancer mimics fluttering and scanning movements, suggesting alertness and awareness. Birds, particularly roosters, are known to ward off darkness with their call at dawn, and in this dance, the bird serves a similar function—dispelling ignorance and inviting the light of wisdom."),
    'Dog': ("Khey", 
            "The Dog mask is a symbol of loyalty, guardianship, and spiritual guidance. The dancer adopts a posture of vigilance, moving with steady and observant steps. In Bhutanese culture, dogs are not only protectors of property but also spiritual beings that guide souls. Within the Peling Nga Cham, the dog ensures the sacred space is protected and may accompany transitions between dance phases."),
    'Horse': ("Ta", 
              "The Horse represents energy, freedom, and the Wind Horse (Lungta), a sacred symbol of good fortune and the bearer of prayers. The dancer performs vigorous, galloping movements that convey vitality and motion. In the Peling Nga Cham, the Horse serves as a messenger between the human and spiritual worlds, uplifting the energy of the performance and guiding positive intentions across realms."),
    'Monkey': ("Tray", 
               "The Monkey is known for its intelligence, curiosity, and mischievousness. The dancer wearing the Monkey mask displays quick, unpredictable movements, often mimicking playful behavior. This figure adds a layer of spontaneity to the dance and can challenge the flow, reminding observers of the monkey mind—restless and ever-changing. The monkey’s role is to provoke awareness and reflection through disruption."),
    'Ox': ("Lang", 
           "The Ox symbolizes strength, patience, and steady progress. In the dance, the Ox dancer moves with deliberate and grounded steps, emphasizing a calm and composed presence. This mask serves as a representation of reliability and endurance, important qualities in both spiritual practice and daily life. The Ox anchors the dance with a sense of order and rhythm, reinforcing stability within the sacred circle."),
    'Pig': ("Phag", 
            "The Pig represents desire, ignorance, and indulgence—qualities often linked to the roots of suffering in Buddhist teachings. The dancer performs heavy, circular movements, reflecting the pig’s attachment to material pleasures. While it may appear comical or even grotesque, the pig plays an important cautionary role in the dance. It serves as a mirror to human tendencies that must be overcome on the path to enlightenment."),
    'Rabbit': ("Yoe", 
               "The Rabbit mask embodies gentleness, caution, and intuitive wisdom. The dancer moves with light, hopping steps, mirroring the rabbit’s peaceful nature and alertness. As a zodiac figure, the rabbit reminds practitioners to tread lightly and listen deeply. Within the dance, it offers a contrast to more aggressive figures, bringing a sense of calm and serenity to the overall performance."),
    'Rat': ("Jewa", 
            "The Rat mask represents intelligence, agility, and foresight. In the Peling Nga Cham, the dancer wearing this mask performs quick and nimble movements that reflect the rat’s cunning and adaptability. As the first animal in the zodiac cycle, the rat leads the dance with alert energy, symbolizing strategic thinking and initiative. Its role is to initiate movement and draw attention to the subtle intelligence found in nature."),
    'Sheep': ("Lu", 
              "The Sheep mask symbolizes compassion, harmony, and communal peace. Gentle and composed, the dancer moves gracefully, creating an atmosphere of kindness and cooperation. In Bhutanese Buddhist culture, the sheep represents the softer qualities of the heart. Within the dance, it helps to balance the intensity of other figures, embodying the nurturing and peaceful aspects of the zodiac cycle."),
    'Snake': ("Drue", 
              "The Snake symbolizes transformation, intuition, and hidden knowledge. The dancer wearing this mask performs sinuous, flowing movements to express the snake’s ability to move silently and unpredictably. As a creature associated with cycles of rebirth, the snake’s presence in the dance highlights themes of inner change and spiritual renewal. It encourages reflection on the shedding of ignorance and attachment."),
    'Tiger': ("Tag", 
              "The Tiger mask signifies courage, power, and protection. It is one of the more imposing masks in the dance, and the performer channels fierce energy through bold, dynamic strides and strong postures. In Bhutanese tradition, the tiger is a guardian of the Dharma and sacred spaces. In Peling Nga Cham, the tiger safeguards the ceremonial ground and awakens the spiritual force among the performers and spectators."),
    'atsara': ("Atsara", 
               "The Atsara combines the spirits of the sacred and the profane, wit and wisdom, humour and responsibility. He uses his pranks to help his audiences not only to forget their worries and problems but also to prod them to overcome their sense of self-importance, hypocrisy and false propriety."),
    'dragon': ("Druk", 
               "The Dragon is one of the most revered symbols in Bhutanese culture, representing spiritual power, authority, and enlightenment. The Dragon mask is used to depict sweeping, majestic movements that mimic wind and thunder. In the Peling Nga Cham, the dragon commands the space with an otherworldly presence, serving as a bridge between the human and divine realms. It is a spiritual awakener, drawing attention to the sacredness of the ritual."),
}

# Inverse mapping from class indices to names
class_idx_to_name = {
    0: 'Bird',
    1: 'Dog',
    2: 'Horse',
    3: 'Monkey',
    4: 'Ox',
    5: 'Pig',
    6: 'Rabbit',
    7: 'Rat',
    8: 'Sheep',
    9: 'Snake',
    10: 'Tiger',
    11: 'atsara',
    12: 'dragon'
}

def home(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Save the uploaded image
            file_obj = request.FILES['image']
            fs = FileSystemStorage()
            file_path = fs.save(file_obj.name, file_obj)
            file_url = fs.url(file_path)

            # Load and preprocess the image
            img = Image.open(fs.path(file_path)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)

            # Make the prediction
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                max_prob, predicted_class_idx = torch.max(probabilities, 1)

            # Confidence threshold
            threshold = 0.80
            if max_prob.item() < threshold:
                class_name = "No masks of this type found"
                description = "The model is not confident in its prediction for the uploaded image."
            else:
                # Map predicted index to class name and description
                predicted_class_name = class_idx_to_name.get(predicted_class_idx.item(), "Unknown")
                class_name, description = class_descriptions.get(predicted_class_name, ("Unknown", "No description available"))

            # Prepare context for rendering
            context = {
                'file_url': file_url,
                'predicted_class_name': class_name,
                'description': description,
                'confidence': f"{max_prob.item() * 100:.2f}%"
            }

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            context = {'error': f"An error occurred: {str(e)}"}

        return render(request, 'predictor/home.html', context)

    return render(request, 'predictor/home.html')
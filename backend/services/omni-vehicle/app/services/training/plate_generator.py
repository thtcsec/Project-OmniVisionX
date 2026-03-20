"""
Synthetic Vietnamese License Plate Generator v4 - REALISTIC EDITION
Generate AUTHENTIC Vietnamese license plates with correct format:
- Motorcycle: 2-line format (29-A1 / 234.56)
- Car: 1-line format (29A-12345)
- Physical augmentations for training LPRv3
Based on Vietnamese-License-Plate-Generator (correct format)
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from pathlib import Path
from typing import Tuple, List, Optional
import os
import math

# Vietnamese plate FORMAT (CORRECT!)
"""
Motorcycle (2 lines):
  Line 1: 29-A1  (province code + series)
  Line 2: 234.56 (sequence with DOT)
  Size: 203x160mm (square)

Car (1 line):
  Single: 29A-12345 (province + series + sequence)
  Size: 340x110mm (rectangle)
"""

# Path to Vietnamese plate font and backgrounds
FONT_PATH = "Vietnamese-License-Plate-Generator/MyFont-Regular_ver3.otf"
# Try multiple possible background paths
BG_SQUARE_OPTIONS = [
    "Vietnamese-License-Plate-Generator/background/bg2.jpg",
    "Vietnamese-License-Plate-Generator/background/square_2.jpg",
    "Vietnamese-License-Plate-Generator/background/square_1.jpg",
]
BG_RECT_OPTIONS = [
    "Vietnamese-License-Plate-Generator/background/bg1.jpg",
    "Vietnamese-License-Plate-Generator/background/rec_1.jpg",
    "Vietnamese-License-Plate-Generator/background/rec_2.jpg",
]

VIETNAM_PROVINCES = [
    '11', '12', '14', '15', '16', '17', '18', '19',  # Cao Bang -> Thai Nguyen
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',  # Lang Son -> Ba Ria
    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',  # Binh Duong -> Ha Noi
    '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',  # Ha Tinh -> Dak Lak
    '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',  # Lao Cai -> Quang Nam
    '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',  # Quang Ngai -> Tien Giang
    '70', '71', '72', '73', '74', '75', '76', '77', '78', '79',  # Tra Vinh -> TPHCM
    '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',  # Long An -> Vinh Long
    '90', '91', '92', '93', '94', '95', '96', '97', '98', '99',  # Bac Lieu -> Ca Mau
]

# Series letters (avoid confusing chars: I, O, Q)
SERIES_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

class VNPlate:
    """Vietnamese License Plate with CORRECT format"""
    
    def __init__(self, province: str, series: str, number: str, vehicle_type: str = 'motorcycle'):
        self.province = province  # 29
        self.series = series      # A1 or AB
        self.number = number      # 23456 (5-6 digits)
        self.vehicle_type = vehicle_type
    
    @property
    def is_2line(self) -> bool:
        """Motorcycle/truck = 2 lines, Car = 1 line"""
        return self.vehicle_type in ['motorcycle', 'truck']
    
    @property
    def line1(self) -> str:
        """Line 1: 29-A1 (for 2-line plates)"""
        return f"{self.province}-{self.series}"
    
    @property
    def line2(self) -> str:
        """Line 2: 234.56 (for 2-line plates, with DOT after 3rd digit)"""
        if len(self.number) >= 5:
            # Insert dot after 3rd digit: 12345 -> 123.45
            return f"{self.number[:3]}.{self.number[3:]}"
        else:
            return self.number
    
    @property
    def single_line(self) -> str:
        """Single line: 29A-12345 (for 1-line car plates)"""
        return f"{self.province}{self.series}-{self.number}"
    
    @property
    def plate_string(self) -> str:
        """Full plate string for label"""
        if self.is_2line:
            return f"{self.line1}/{self.line2}"
        else:
            return self.single_line
    
    @classmethod
    def random(cls, vehicle_type='motorcycle') -> 'VNPlate':
        """Generate random Vietnamese plate"""
        province = random.choice(VIETNAM_PROVINCES)
        
        # Series: 1-2 letters
        if random.random() < 0.7:
            # Single letter (70%)
            series = random.choice(SERIES_LETTERS)
        else:
            # Double letter (30%)
            series = random.choice(SERIES_LETTERS) + random.choice(SERIES_LETTERS)
        
        # Number: 5-6 digits
        num_digits = random.choice([5, 6])
        number = ''.join(random.choices('0123456789', k=num_digits))
        
        return cls(province, series, number, vehicle_type)


class PlateGenerator:
    """Generate REALISTIC Vietnamese license plates"""
    
    def __init__(self, font_path: Optional[str] = None, procedural_noise: Optional[dict] = None):
        """Initialize with Vietnamese plate font"""
        if font_path is None:
            font_path = FONT_PATH
        
        self.font_path = font_path
        self.procedural_noise = procedural_noise or {}
        
        # Check if font exists
        if not os.path.exists(font_path):
            print(f"⚠️ Font not found: {font_path}")
            print(f"   Using fallback font (will look FAKE)")
            self.font_path = None

    def _apply_glare(self, img: np.ndarray, intensity_range: Tuple[int, int] = (80, 180)) -> np.ndarray:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cx = np.random.randint(int(w * 0.2), int(w * 0.8))
        cy = np.random.randint(int(h * 0.2), int(h * 0.8))
        radius = np.random.randint(max(3, int(min(h, w) * 0.08)), max(5, int(min(h, w) * 0.18)))
        intensity = np.random.uniform(*intensity_range)
        cv2.circle(mask, (cx, cy), radius, intensity, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.6)
        glare = np.repeat(mask[:, :, None], 3, axis=2)
        return np.clip(img.astype(np.float32) + glare, 0, 255).astype(np.uint8)

    def _apply_rain_streaks(self, img: np.ndarray, density: int = 140) -> np.ndarray:
        h, w = img.shape[:2]
        streaks = np.zeros((h, w), dtype=np.float32)
        for _ in range(density):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(max(6, h // 16), max(12, h // 6))
            thickness = np.random.randint(1, 2 + (w // 200))
            angle = np.random.uniform(-15, 15)
            dx = int(math.cos(math.radians(angle)) * length)
            dy = int(math.sin(math.radians(angle)) * length) + length
            cv2.line(streaks, (x, y), (x + dx, y + dy), 255, thickness)
        streaks = cv2.GaussianBlur(streaks, (0, 0), sigmaX=2)
        rain = np.repeat(streaks[:, :, None], 3, axis=2)
        return np.clip(img.astype(np.float32) + rain * 0.25, 0, 255).astype(np.uint8)

    def _apply_mud_mask(self, img: np.ndarray, blobs: int = 3) -> np.ndarray:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(blobs):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            rx = np.random.randint(max(6, w // 12), max(10, w // 5))
            ry = np.random.randint(max(4, h // 10), max(8, h // 4))
            angle = np.random.uniform(0, 180)
            cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)
        brown = np.array([60, 50, 40], dtype=np.float32)[None, None, :]
        overlay = brown * mask[:, :, None]
        mixed = img.astype(np.float32) * (1 - mask[:, :, None]) + overlay
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def _apply_motion_blur(self, img: np.ndarray, magnitude: int = 12) -> np.ndarray:
        k = max(3, int(magnitude))
        angle = np.random.uniform(0, 180)
        kernel = np.zeros((k, k), dtype=np.float32)
        cv2.line(kernel, (k // 2, 0), (k // 2, k - 1), 1.0, 1)
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
        kernel = kernel / max(kernel.sum(), 1e-6)
        return cv2.filter2D(img, -1, kernel)
    
    def generate_plate_image(self, plate: VNPlate) -> Image.Image:
        """Generate REALISTIC plate image with correct format"""
        
        if plate.is_2line:
            return self._generate_2line_plate(plate)
        else:
            return self._generate_1line_plate(plate)
    
    def _generate_1line_plate(self, plate: VNPlate) -> Image.Image:
        """Generate 1-line plate (Car): 29A-12345 - EXACTLY like original code"""
        # Load background (try multiple paths)
        im = None
        for bg_path in BG_RECT_OPTIONS:
            if os.path.exists(bg_path):
                try:
                    im = Image.open(bg_path)
                    break
                except:
                    continue
        
        if im is None:
            # Fallback: Create white background
            im = Image.new('RGB', (450, 110), (255, 255, 255))
        
        # Resize based on plate length (EXACTLY like original)
        plate_text = plate.single_line
        if len(plate_text.replace('-', '').replace('.', '')) > 7:
            im = im.resize((450, 110))
        else:
            im = im.resize((400, 110))
        
        width, height = im.size
        draw = ImageDraw.Draw(im)
        
        # Load font (EXACTLY like original: size 108)
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, 108)
            else:
                # Try to find font in current dir
                local_font = "MyFont-Regular_ver3.otf"
                if os.path.exists(local_font):
                    font = ImageFont.truetype(local_font, 108)
                else:
                    font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = font.getbbox(plate_text)
        textsize = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        textX = int((width - textsize[0]) / 2)
        textY = int((height - textsize[1]) / 2)
        
        # Random dark color (EXACTLY like original: 0-19)
        fill = tuple(np.random.randint(20, size=3))
        
        draw.text((textX, textY), plate_text, font=font, fill=fill)
        
        return im
    
    def _generate_2line_plate(self, plate: VNPlate, size: Tuple[int, int] = (480, 400), margin: int = 10) -> Image.Image:
        """Generate 2-line plate (Motorcycle): 29-A1 / 234.56 - EXACTLY like original code"""
        # Load background (try multiple paths)
        im = None
        for bg_path in BG_SQUARE_OPTIONS:
            if os.path.exists(bg_path):
                try:
                    im = Image.open(bg_path)
                    break
                except:
                    continue
        
        if im is None:
            # Fallback: Create white background
            im = Image.new('RGB', size, (255, 255, 255))
        
        im = im.resize(size)
        width, height = im.size
        draw = ImageDraw.Draw(im)
        
        # Load font (EXACTLY like original: size 180)
        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, 180)
            else:
                # Try to find font in current dir
                local_font = "MyFont-Regular_ver3.otf"
                if os.path.exists(local_font):
                    font = ImageFont.truetype(local_font, 180)
                else:
                    font = ImageFont.truetype("arial.ttf", 100)
        except:
            font = ImageFont.load_default()
        
        line1 = plate.line1  # 29-A1
        line2 = plate.line2  # 234.56
        
        # Calculate positions (EXACTLY like original code)
        bbox1 = font.getbbox(line1)
        textsize1 = (bbox1[2] - bbox1[0], bbox1[3] - bbox1[1])
        textX1 = int((width - textsize1[0]) / 2)
        textY1 = int((height/2 - textsize1[1]) / 2) + margin
        
        bbox2 = font.getbbox(line2)
        textsize2 = (bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])
        textX2 = int((width - textsize2[0]) / 2)
        textY2 = int(height/2 + (height/2 - textsize2[1]) / 2) - margin/2
        
        # Colors (EXACTLY like original)
        fill = tuple(np.random.randint(20, size=3))  # Dark text (0-19)
        shadow = tuple(np.random.randint(200, 255, size=3))  # Light shadow (200-254)
        direction = tuple(np.random.randint(-3, 3, size=2))  # Shadow offset (-3 to 2)
        
        # ALWAYS draw shadow FIRST (like original code - NOT random!)
        draw.text((textX1 + direction[0], textY1 + direction[1]), line1, font=font, fill=shadow)
        draw.text((textX2 + direction[0], textY2 + direction[1]), line2, font=font, fill=shadow)
        
        # Then draw main text on top
        draw.text((textX1, textY1), line1, font=font, fill=fill)
        draw.text((textX2, textY2), line2, font=font, fill=fill)
        
        return im
    
    def specular_glare(self, img: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """
        Add specular glare (lens flare) effect
        Simulates sunlight/headlight reflection on plate
        """
        h, w = img.shape[:2]
        output = img.copy().astype(np.float32)
        
        # Random glare positions (usually corners or center)
        glare_x = random.randint(int(w*0.2), int(w*0.8))
        glare_y = random.randint(int(h*0.2), int(h*0.8))
        
        # Glare radius
        radius = random.randint(20, 80)
        
        # Create radial gradient for glare
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - glare_x)**2 + (yy - glare_y)**2)
        
        glare_mask = np.exp(-(dist**2) / (2 * radius**2))
        glare_mask = glare_mask * intensity
        
        # Apply glare (brighten affected area)
        for c in range(3):
            output[:, :, c] = output[:, :, c] * (1 - glare_mask * 0.7) + 255 * glare_mask
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def ir_washout(self, img: np.ndarray, intensity: float = 0.2) -> np.ndarray:
        """
        IR washout effect - simulates thermal/IR camera saturation
        Common in night/low-light CCTV footage
        """
        h, w = img.shape[:2]
        output = img.astype(np.float32)
        
        # Create thermal washout pattern
        if len(img.shape) == 3:
            # Color IR washout - often reddish/pinkish tint
            output[:, :, 2] = np.clip(output[:, :, 2] * (1 + intensity * 0.5), 0, 255)  # Red channel boost
            output[:, :, 1] = np.clip(output[:, :, 1] * (1 + intensity * 0.3), 0, 255)  # Green channel slight boost
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def radial_distortion(self, img: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """
        Radial distortion - simulates fisheye/wide-angle lens
        Common in wide-angle CCTV cameras
        """
        h, w = img.shape[:2]
        
        # Create distortion map
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize to [-1, 1]
        x = (X - w/2) / (w/2)
        y = (Y - h/2) / (h/2)
        
        # Apply barrel distortion
        r = np.sqrt(x**2 + y**2)
        distortion = 1 + strength * r**2
        
        x_distorted = x * distortion
        y_distorted = y * distortion
        
        # Map back to image coordinates
        X_distorted = (x_distorted * w/2 + w/2).astype(np.float32)
        Y_distorted = (y_distorted * h/2 + h/2).astype(np.float32)
        
        # Remap image
        distorted = cv2.remap(img, X_distorted, Y_distorted, cv2.INTER_LINEAR)
        
        return distorted
    
    def motion_blur(self, img: np.ndarray, kernel_size: int = 5, angle: float = None) -> np.ndarray:
        """Motion blur - simulates moving vehicle/camera"""
        if angle is None:
            angle = random.uniform(0, 180)
        
        # Create motion kernel
        kernel = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1)
        kernel = cv2.warpAffine(
            np.eye(kernel_size),
            kernel,
            (kernel_size, kernel_size)
        )
        kernel = kernel / kernel.sum()
        
        # Apply blur
        blurred = cv2.filter2D(img, -1, kernel)
        
        return blurred
    
    def gaussian_blur(self, img: np.ndarray, sigma: float = 0.5) -> np.ndarray:
        """Gaussian blur - simulates focus issues"""
        kernel_size = int(sigma * 4) * 2 + 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    def rain_drops(self, img: np.ndarray, drop_count: int = 10) -> np.ndarray:
        """Rain drops on lens - circular artifacts"""
        output = img.copy()
        h, w = img.shape[:2]
        
        for _ in range(drop_count):
            # Random position
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            radius = random.randint(2, 8)
            
            # Create semi-transparent water drop
            cv2.circle(output, (x, y), radius, (200, 200, 255), -1)
            cv2.circle(output, (x, y), radius, (100, 100, 150), 2)
        
        return output
    
    def weather_noise(self, img: np.ndarray, noise_type: str = 'gaussian', intensity: float = 0.1) -> np.ndarray:
        """Add weather-related noise"""
        h, w = img.shape[:2]
        output = img.astype(np.float32)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 255 * intensity, img.shape)
        elif noise_type == 'salt_pepper':
            noise = np.random.choice([0, 255, 0], img.shape)
            mask = np.random.random(img.shape) < intensity
            noise[~mask] = 0
        else:
            noise = 0
        
        output = output + noise
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def augment_plate(self, plate: VNPlate, num_variants: int = 5) -> List[Tuple[np.ndarray, str]]:
        """
        Generate multiple augmented versions of single plate
        Returns: List of (image_array, plate_string) tuples
        
        Plate string format:
        - 2-line: "29-A1/234.56"
        - 1-line: "29A-12345"
        """
        clean_img = self.generate_plate_image(plate)
        
        # Convert PIL to numpy (RGB to BGR for OpenCV)
        img_np = cv2.cvtColor(np.array(clean_img), cv2.COLOR_RGB2BGR)
        
        variants = []
        
        # Original clean version (15% chance)
        if random.random() < 0.15:
            variants.append((img_np, plate.plate_string))
        
        # Augmentation functions
        glare_prob = float(self.procedural_noise.get("glare_prob", 0.0) or 0.0)
        rain_prob = float(self.procedural_noise.get("rain_prob", 0.0) or 0.0)
        mud_prob = float(self.procedural_noise.get("mud_prob", 0.0) or 0.0)
        motion_blur_prob = float(self.procedural_noise.get("motion_blur_prob", 0.0) or 0.0)
        aug_functions = [
            lambda img: self.specular_glare(img, intensity=random.uniform(0.1, 0.4)),
            lambda img: self.ir_washout(img, intensity=random.uniform(0.05, 0.2)),
            lambda img: self.radial_distortion(img, strength=random.uniform(0.02, 0.1)),
            lambda img: self.motion_blur(img, kernel_size=random.choice([3, 5, 7])),
            lambda img: self.gaussian_blur(img, sigma=random.uniform(0.3, 1.5)),
            lambda img: self.rain_drops(img, drop_count=random.randint(3, 15)),
            lambda img: self.weather_noise(img, noise_type='gaussian', intensity=random.uniform(0.01, 0.05)),
            lambda img: self._apply_glare(img) if random.random() < glare_prob else img,
            lambda img: self._apply_rain_streaks(img) if random.random() < rain_prob else img,
            lambda img: self._apply_mud_mask(img) if random.random() < mud_prob else img,
            lambda img: self._apply_motion_blur(img, magnitude=random.randint(8, 18)) if random.random() < motion_blur_prob else img,
        ]
        
        # Combine random augmentations
        for _ in range(num_variants):
            aug_img = img_np.copy()
            
            # Apply 1-3 random augmentations
            num_augs = random.randint(1, 3)
            for _ in range(num_augs):
                aug_func = random.choice(aug_functions)
                aug_img = aug_func(aug_img)
            
            variants.append((aug_img, plate.plate_string))
        
        return variants


class DatasetGenerator:
    """Generate REALISTIC Vietnamese plate dataset"""
    
    def __init__(self, output_dir: str = "dataset/vn_plates_realistic", procedural_noise: Optional[dict] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata_file = self.output_dir / "metadata.txt"
        
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        
        self.generator = PlateGenerator(procedural_noise=procedural_noise)
        self.image_count = 0
        
        print("=" * 60)
        print("🇻🇳 VIETNAMESE LICENSE PLATE GENERATOR v4 - REALISTIC")
        print("=" * 60)
        print(f"📁 Output: {self.output_dir}")
        print(f"🎨 Format: ")
        print(f"   - Motorcycle (2-line): 29-A1 / 234.56")
        print(f"   - Car (1-line): 29A-12345")
        print(f"🔧 Augmentations:")
        print(f"   ✅ Glare, IR washout, Distortion")
        print(f"   ✅ Motion blur, Gaussian blur")
        print(f"   ✅ Rain drops, Weather noise")
        print("=" * 60)
    
    def generate_dataset(self, num_plates: int = 50000, variants_per_plate: int = 5):
        """
        Generate synthetic dataset
        
        Args:
            num_plates: Number of unique plates (default 50k)
            variants_per_plate: Augmented versions per plate (default 5)
        
        Total images = num_plates * variants_per_plate (e.g., 50k * 5 = 250k images)
        """
        total_images = num_plates * variants_per_plate
        print(f"\n🚀 Generating {num_plates:,} unique plates × {variants_per_plate} variants = {total_images:,} total images...")
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            f.write("image_name\tplate_string\tvehicle_type\tis_2line\tvariant_idx\n")
        
        for plate_idx in range(num_plates):
            # Random vehicle type (70% motorcycle, 25% car, 5% truck)
            rand = random.random()
            if rand < 0.70:
                vehicle_type = 'motorcycle'
            elif rand < 0.95:
                vehicle_type = 'car'
            else:
                vehicle_type = 'truck'
            
            # Generate random plate
            plate = VNPlate.random(vehicle_type=vehicle_type)
            
            # Generate augmented variants
            variants = self.generator.augment_plate(plate, num_variants=variants_per_plate)
            
            for variant_idx, (img_array, plate_str) in enumerate(variants):
                # Save image
                img_name = f"{self.image_count:06d}.jpg"
                img_path = self.images_dir / img_name
                
                cv2.imwrite(str(img_path), img_array)
                
                # Save label (plate string)
                label_name = f"{self.image_count:06d}.txt"
                label_path = self.labels_dir / label_name
                
                with open(label_path, 'w', encoding='utf-8') as lf:
                    lf.write(plate_str)
                
                # Log metadata
                with open(self.metadata_file, 'a', encoding='utf-8') as f:
                    is_2line = '2line' if plate.is_2line else '1line'
                    f.write(f"{img_name}\t{plate_str}\t{vehicle_type}\t{is_2line}\t{variant_idx}\n")
                
                self.image_count += 1
            
            # Progress update every 1000 plates
            if (plate_idx + 1) % 1000 == 0:
                percent = ((plate_idx + 1) / num_plates) * 100
                print(f"  📊 Progress: {plate_idx+1:,}/{num_plates:,} plates ({percent:.1f}%) → {self.image_count:,} images generated")
        
        print(f"\n{'='*60}")
        print(f"✅ Dataset generation COMPLETE!")
        print(f"📊 Statistics:")
        print(f"   - Unique plates: {num_plates:,}")
        print(f"   - Total images: {self.image_count:,}")
        print(f"   - Variants per plate: {variants_per_plate}")
        print(f"📁 Location: {self.output_dir}")
        print(f"   - Images: {self.images_dir}")
        print(f"   - Labels: {self.labels_dir}")
        print(f"{'='*60}")
        
        return self.image_count


if __name__ == "__main__":
    # Quick test: Generate 100 plates with 5 variants each = 500 images
    print("\n🧪 TEST MODE: Generating 100 sample plates...\n")
    gen = DatasetGenerator(output_dir="dataset/test_realistic")
    gen.generate_dataset(num_plates=100, variants_per_plate=5)
    
    print("\n💡 To generate FULL dataset (200k images):")
    print("   gen = DatasetGenerator()")
    print("   gen.generate_dataset(num_plates=40000, variants_per_plate=5)")

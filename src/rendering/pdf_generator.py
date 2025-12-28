"""PDF generation for trajectory visualization output."""

import cv2
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
from typing import Optional, List


class PDFGenerator:
    """Generates PDF output with annotated pool table image."""

    def __init__(self, page_size='letter'):
        """Initialize PDF generator.

        Args:
            page_size: Page size ('letter' or 'A4')
        """
        self.page_size = letter if page_size == 'letter' else A4

    def create_pdf(
        self,
        output_path: str,
        annotated_image: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        title: str = "GamePigeon 8 Ball - Trajectory Prediction",
        metadata: Optional[dict] = None
    ):
        """Create PDF with annotated image.

        Args:
            output_path: Path to save PDF
            annotated_image: Annotated image with trajectories
            original_image: Optional original image (if provided, creates two-page PDF)
            title: PDF title
            metadata: Optional metadata dictionary
        """
        # Create canvas
        c = canvas.Canvas(output_path, pagesize=self.page_size)
        page_width, page_height = self.page_size

        # Set metadata
        c.setTitle(title)
        c.setAuthor("GamePigeon Pool Predictor")
        c.setSubject("Pool Ball Trajectory Prediction")
        c.setCreator("GamePigeon Pool Predictor v1.0")

        # Add title page header
        self._add_header(c, title, page_width, page_height)

        # Convert annotated image to PIL format
        annotated_pil = self._cv2_to_pil(annotated_image)

        # Calculate image dimensions to fit on page
        img_x, img_y, img_width, img_height = self._calculate_image_placement(
            annotated_pil, page_width, page_height, margin=50, title_offset=100
        )

        # Draw annotated image
        c.drawImage(
            ImageReader(annotated_pil),
            img_x, img_y,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True
        )

        # Add metadata text if provided
        if metadata:
            self._add_metadata(c, metadata, page_width, img_y - 10)

        # Add footer
        self._add_footer(c, page_width, page_height)

        # Finish first page
        c.showPage()

        # If original image provided, add it as second page
        if original_image is not None:
            self._add_header(c, "Original Screenshot", page_width, page_height)

            original_pil = self._cv2_to_pil(original_image)
            img_x, img_y, img_width, img_height = self._calculate_image_placement(
                original_pil, page_width, page_height, margin=50, title_offset=100
            )

            c.drawImage(
                ImageReader(original_pil),
                img_x, img_y,
                width=img_width,
                height=img_height,
                preserveAspectRatio=True
            )

            self._add_footer(c, page_width, page_height)
            c.showPage()

        # Save PDF
        c.save()

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL Image.

        Args:
            cv2_image: OpenCV BGR image

        Returns:
            PIL Image in RGB
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    def _calculate_image_placement(
        self,
        pil_image: Image.Image,
        page_width: float,
        page_height: float,
        margin: float = 50,
        title_offset: float = 100
    ) -> tuple:
        """Calculate image placement to fit on page.

        Args:
            pil_image: PIL Image
            page_width: Page width
            page_height: Page height
            margin: Margin around image
            title_offset: Space reserved for title at top

        Returns:
            Tuple of (x, y, width, height) for image placement
        """
        # Get image dimensions
        img_width, img_height = pil_image.size

        # Calculate available space
        available_width = page_width - 2 * margin
        available_height = page_height - title_offset - margin

        # Calculate scale factor to fit image
        scale_x = available_width / img_width
        scale_y = available_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale

        # Calculate scaled dimensions
        scaled_width = img_width * scale
        scaled_height = img_height * scale

        # Center image horizontally
        x = (page_width - scaled_width) / 2

        # Position from top, leaving space for title
        y = page_height - title_offset - scaled_height - margin

        return x, y, scaled_width, scaled_height

    def _add_header(self, c: canvas.Canvas, title: str, page_width: float, page_height: float):
        """Add header with title to page.

        Args:
            c: Canvas object
            title: Title text
            page_width: Page width
            page_height: Page height
        """
        c.setFont("Helvetica-Bold", 16)
        title_y = page_height - 40
        c.drawCentredString(page_width / 2, title_y, title)

        # Add timestamp
        c.setFont("Helvetica", 10)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.drawCentredString(page_width / 2, title_y - 20, f"Generated: {timestamp}")

    def _add_footer(self, c: canvas.Canvas, page_width: float, page_height: float):
        """Add footer to page.

        Args:
            c: Canvas object
            page_width: Page width
            page_height: Page height
        """
        c.setFont("Helvetica", 8)
        footer_text = "Generated by GamePigeon Pool Predictor"
        c.drawCentredString(page_width / 2, 30, footer_text)

    def _add_metadata(self, c: canvas.Canvas, metadata: dict, page_width: float, y_position: float):
        """Add metadata text to page.

        Args:
            c: Canvas object
            metadata: Dictionary of metadata
            page_width: Page width
            y_position: Y position to start metadata
        """
        c.setFont("Helvetica", 9)

        text_lines = []
        if 'balls_detected' in metadata:
            text_lines.append(f"Balls detected: {metadata['balls_detected']}")
        if 'cue_angle' in metadata:
            text_lines.append(f"Shot angle: {metadata['cue_angle']:.1f}°")
        if 'collisions' in metadata:
            text_lines.append(f"Predicted collisions: {metadata['collisions']}")

        y = y_position
        for line in text_lines:
            c.drawCentredString(page_width / 2, y, line)
            y -= 15

    def create_simple_pdf(self, output_path: str, image: np.ndarray):
        """Create simple single-page PDF with just the image.

        Args:
            output_path: Path to save PDF
            image: Image to include
        """
        self.create_pdf(
            output_path,
            annotated_image=image,
            original_image=None,
            title="Pool Ball Trajectory Prediction"
        )

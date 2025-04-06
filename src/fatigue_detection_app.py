from __future__ import annotations
import sys
import time
import enum
from pathlib import Path
from typing import Any, Tuple, Dict, Optional
from contextlib import contextmanager

import cv2
import gradio as gr
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from src.config import AppConfig, AlgorithmType
from src.core.detection.analyzer import FatigueAnalyzer

# Plain ASCII art without formatting markers
APP_ASCII = """
███████╗ █████╗ ████████╗██╗ ██████╗ ██╗   ██╗███████╗
██╔════╝██╔══██╗╚══██╔══╝██║██╔════╝ ██║   ██║██╔════╝
█████╗  ███████║   ██║   ██║██║  ███╗██║   ██║█████╗  
██╔══╝  ██╔══██║   ██║   ██║██║   ██║██║   ██║██╔══╝  
██║     ██║  ██║   ██║   ██║╚██████╔╝╚██████╔╝███████╗
╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝
⠀⠀⠀⠀⠀⠀⢀⡴⠁⠀⠀⣠⠎⠀⠀⠀⣴⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢠⣿⣧⡀⠀⢰⣿⣄⠀⠀⣾⣿⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠙⠻⣿⣷⡌⠛⢿⣿⣦⠈⠛⢿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⡸⠟⠀⠀⢀⠿⠃⠀⠀⠀⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣀⣀⣈⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⠀⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠿⣷⣦⡀⠀⠀⠀
⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠈⠻⣷⡀⠀⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⣿⡇⠀⠀
⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠀⣀⣾⡿⠁⠀⠀
⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⣸⣿⣿⠿⠛⠁⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠠⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⡆⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀
"""

class GradioInterface:
    """Advanced Gradio UI with real-time monitoring and file analysis"""
    
    def __init__(self, analyzer: FatigueAnalyzer):
        self.analyzer = analyzer
        self.console = Console()
        self._setup_interface()
        
    def _setup_interface(self):
        """Setup the Gradio interface components"""
        self.interface = None  # Will be initialized in create_ui

    def create_ui(self):
        """Create the complete UI with tabs for different input methods"""
        webcam_interface = gr.Interface(
            fn=self._process_webcam,
            inputs=gr.Image(sources=["webcam"], type="numpy"),
            outputs=[
                gr.Image(label="Processed Feed"),
                gr.JSON(label="Analysis Results"),
                gr.HTML(label="System Status")
            ],
            live=True,
            title="Live Camera Analysis",
            description="Analyze fatigue in real-time using your webcam. If the camera is not available, please use the File Analysis tab."
        )
        
        file_interface = gr.Interface(
            fn=self._process_file,
            inputs=gr.File(label="Upload Image or Video", file_types=["image", "video"]),
            outputs=[
                gr.Image(label="Processed Result"),
                gr.JSON(label="Analysis Results"),
                gr.HTML(label="System Status")
            ],
            title="File Analysis",
            description="Upload an image or video file for fatigue analysis."
        )
        
        return gr.TabbedInterface(
            [file_interface, webcam_interface],
            ["File Analysis", "Live Camera"]
        )

    def _process_webcam(self, image):
        """Process webcam feed and detect fatigue"""
        if image is None:
            return None, {"error": "No image provided"}, "<p>Waiting for camera feed...</p>"
        
        try:
            # Convert Gradio image to OpenCV format if needed
            if isinstance(image, np.ndarray):
                frame = image.copy()
            else:
                # Handle other image formats if necessary
                return None, {"error": "Unsupported image format"}, "<p>Error: Unsupported image format</p>"
            
            # Analyze the frame
            results = self.analyzer.analyze_frame(frame)
            
            # Draw visualization on the frame
            processed_frame = self._draw_visualization(frame, results)
            
            # Generate status HTML
            status_html = self._generate_status_html(results)
            
            return processed_frame, results, status_html
        except Exception as e:
            logger.error(f"Error processing webcam feed: {str(e)}")
            error_html = f"""<div style='padding: 10px; background-color: #ffeeee; border-radius: 5px; border: 1px solid #ffcccc;'>
                <h3 style='color: #cc0000;'>Camera Error</h3>
                <p>{str(e)}</p>
                <p>Please try using the File Analysis tab instead.</p>
            </div>"""
            return image, {"error": str(e)}, error_html
            
    def _process_file(self, file):
        """Process uploaded image or video file"""
        if file is None:
            return None, {"error": "No file provided"}, "<p>Please upload an image or video file.</p>"
        
        try:
            # Check file type and handle accordingly
            file_path = file.name
            file_ext = Path(file_path).suffix.lower()
            
            # Handle image files
            if file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                # Read image file
                image_data = file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Analyze the frame
                results = self.analyzer.analyze_frame(frame)
                
                # Draw visualization
                processed_frame = self._draw_visualization(frame, results)
                
                # Generate status HTML
                status_html = self._generate_status_html(results)
                
                return processed_frame, results, status_html
                
            # Handle video files
            elif file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
                # For videos, we'll analyze the first few frames
                # Save the file to a temporary location
                temp_path = Path(file_path).name
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                
                # Open the video file
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    return None, {"error": "Could not open video file"}, "<p>Error: Could not open video file</p>"
                
                # Read the first frame
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    return None, {"error": "Could not read video frame"}, "<p>Error: Could not read video frame</p>"
                
                # Analyze the frame
                results = self.analyzer.analyze_frame(frame)
                
                # Draw visualization
                processed_frame = self._draw_visualization(frame, results)
                
                # Generate status HTML
                status_html = self._generate_status_html(results)
                
                # Clean up
                cap.release()
                try:
                    Path(temp_path).unlink()  # Delete temporary file
                except Exception as e:
                    logger.warning(f"Could not delete temporary file: {str(e)}")
                
                return processed_frame, results, status_html
            else:
                return None, {"error": "Unsupported file format"}, "<p>Error: Unsupported file format. Please upload an image or video file.</p>"
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            error_html = f"""<div style='padding: 10px; background-color: #ffeeee; border-radius: 5px; border: 1px solid #ffcccc;'>
                <h3 style='color: #cc0000;'>File Processing Error</h3>
                <p>{str(e)}</p>
            </div>"""
            return None, {"error": str(e)}, error_html
    
    def _draw_visualization(self, frame, results):
        """Draw visualization overlays on the frame"""
        # Add visual indicators based on fatigue detection
        if results.get("fatigue_detected", False):
            # Draw red border for fatigue detection
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            cv2.putText(frame, "FATIGUE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add confidence score
        confidence = results.get("confidence", 0.0)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _generate_status_html(self, results):
        """Generate HTML status display"""
        if "error" in results:
            return f"<p style='color: red;'>Error: {results['error']}</p>"
        
        fatigue_detected = results.get("fatigue_detected", False)
        confidence = results.get("confidence", 0.0)
        
        color = "red" if fatigue_detected else "green"
        status = "FATIGUE DETECTED" if fatigue_detected else "NORMAL"
        
        html = f"""
        <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
            <h3 style='color: {color};'>{status}</h3>
            <p>Confidence: {confidence:.2f}</p>
        </div>
        """
        
        return html
        
    def _run_console_mode(self):
        """Run in console mode as a fallback when GUI is not available"""
        self.console.print("[bold green]Running in console mode[/bold green]")
        self.console.print("[yellow]Please select an option:[/yellow]")
        self.console.print("1. Analyze an image file")
        self.console.print("2. Exit")
        
        choice = input("Enter your choice (1-2): ")
        
        if choice == "1":
            file_path = input("Enter the path to the image file: ")
            try:
                # Check if file exists
                if not Path(file_path).exists():
                    self.console.print(f"[bold red]File not found: {file_path}[/bold red]")
                    return self._run_console_mode()
                
                # Read the image
                frame = cv2.imread(file_path)
                if frame is None:
                    self.console.print(f"[bold red]Could not read image: {file_path}[/bold red]")
                    return self._run_console_mode()
                
                # Analyze the frame
                self.console.print("[bold]Analyzing image...[/bold]")
                results = self.analyzer.analyze_frame(frame)
                
                # Display results in a table
                table = Table(title="Fatigue Analysis Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                table.add_row("Fatigue Detected", "YES" if results.get("fatigue_detected", False) else "NO")
                table.add_row("Confidence", f"{results.get('confidence', 0.0):.2f}")
                
                if "features" in results:
                    for feature, value in results["features"].items():
                        if isinstance(value, dict):
                            for subfeature, subvalue in value.items():
                                table.add_row(f"{feature}.{subfeature}", str(subvalue))
                        else:
                            table.add_row(feature, str(value))
                
                self.console.print(table)
                
                # Ask if user wants to continue
                self.console.print("\n")
                return self._run_console_mode()
                
            except Exception as e:
                self.console.print(f"[bold red]Error analyzing image: {str(e)}[/bold red]")
                return self._run_console_mode()
        elif choice == "2":
            self.console.print("[bold green]Exiting application. Goodbye![/bold green]")
            return
        else:
            self.console.print("[bold red]Invalid choice. Please try again.[/bold red]")
            return self._run_console_mode()
    
    def launch(self):
        """Start the Gradio interface"""
        try:
            self.create_ui().launch(server_name="0.0.0.0", share=False)
        except Exception as e:
            logger.error(f"Error launching Gradio interface: {str(e)}")
            self.console.print(f"[bold red]Error launching interface: {str(e)}[/bold red]")
            self.console.print("[yellow]The application will continue to run in console mode.[/yellow]")
            # Fallback to console mode if Gradio fails to launch
            self._run_console_mode()

def main():
    """Application entry point"""
    # Print ASCII art without rich text formatting
    print(APP_ASCII)
    config = AppConfig()
    analyzer = FatigueAnalyzer(config)
    
    console = Console()
    # Use a simple print statement instead of a blocking status context manager
    print("Initializing Fatigue Detection App...")
    interface = GradioInterface(analyzer)
    interface.launch()

if __name__ == "__main__":
    main()
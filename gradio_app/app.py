#!/usr/bin/env python3
"""
Optimized Book Cover Generator - Gradio Interface
DreamBooth trained Stable Diffusion model for generating book covers
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import os
import gc
from functools import lru_cache

# Global pipeline variable
pipeline = None

def load_model():
    """Load the DreamBooth trained model with optimizations"""
    global pipeline
    
    try:
        print("Loading DreamBooth model with optimizations...")
        
        # Try to load from local model directory
        model_path = "./model"
        
        # Determine dtype based on device availability
        if torch.cuda.is_available():
            dtype = torch.float16  # GPU can handle float16
        else:
            dtype = torch.float32  # CPU works better with float32
            
        if os.path.exists(model_path):
            # Try loading with different configurations
            try:
                # First try with safetensors and fp16 variant (only on GPU)
                if torch.cuda.is_available():
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                        variant="fp16"
                    )
                    print("✅ Loaded with safetensors fp16 variant")
                else:
                    raise Exception("Skip fp16 variant on CPU")
            except:
                try:
                    # Try with safetensors but no variant
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True
                    )
                    print("✅ Loaded with safetensors")
                except:
                    # Fallback to standard loading
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                    print("✅ Loaded with standard method")
        else:
            # Fallback to base model if custom model not found
            print("Custom model not found, using base model...")
            pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
        
        # Use faster scheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        # Enable memory efficient attention if available
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory efficient attention enabled")
        except:
            print("⚠️ XFormers not available, using default attention")
        
        # Move to device with proper dtype handling
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            
            # Enable memory efficient attention for CUDA
            try:
                pipeline.enable_attention_slicing(1)  # Reduces VRAM usage
                pipeline.enable_cpu_offload()  # Offload to CPU when not in use
                print("✅ Memory optimizations enabled")
            except:
                print("⚠️ Some optimizations not available")
                
            print("Model loaded on GPU with optimizations")
        else:
            # CPU optimizations - ensure everything is float32
            try:
                pipeline = pipeline.to("cpu", torch.float32)
                pipeline.enable_attention_slicing("auto")
                print("✅ CPU optimizations enabled (float32)")
            except Exception as cpu_opt_error:
                print(f"⚠️ CPU optimizations failed: {cpu_opt_error}")
            
            print("Model loaded on CPU with float32")
            
        # Warm up the pipeline with a small generation (skip on CPU for speed)
        if torch.cuda.is_available():
            print("Warming up model...")
            warmup_prompt = "a simple book cover"
            with torch.no_grad():
                _ = pipeline(
                    warmup_prompt, 
                    num_inference_steps=1, 
                    height=64, 
                    width=64,
                    guidance_scale=1.0
                )
            print("✅ Model warmed up")
        else:
            print("✅ Skipping warmup on CPU for faster startup")
            
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@lru_cache(maxsize=3)  # Cache fonts to avoid reloading
def load_fonts():
    """Load fonts with caching"""
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        author_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 30)
        return title_font, author_font, subtitle_font
    except:
        try:
            default_font = ImageFont.load_default()
            return default_font, default_font, default_font
        except:
            return None, None, None

def add_text_to_cover(image, title, author="", subtitle=""):
    """Optimized text overlay function"""
    
    if image is None:
        return None
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Create a copy to work with
    img = image.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    
    # Load cached fonts
    title_font, author_font, subtitle_font = load_fonts()
    
    if not title_font:  # Skip text if fonts failed to load
        return img
    
    # Optimized text positioning
    title_y = H // 4
    subtitle_y = title_y + 80
    author_y = H - 100
    
    # Simplified shadow effect (faster)
    def draw_text_with_shadow(pos, text, font, fill_color="white"):
        x, y = pos
        # Single shadow for speed
        draw.text((x-2, y-2), text, font=font, fill="black")
        draw.text(pos, text, font=font, fill=fill_color)
    
    # Draw title
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (W - title_width) // 2
        draw_text_with_shadow((title_x, title_y), title, title_font, "white")
    
    # Draw subtitle
    if subtitle:
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (W - subtitle_width) // 2
        draw_text_with_shadow((subtitle_x, subtitle_y), subtitle, subtitle_font, "#f0f0f0")
    
    # Draw author
    if author:
        author_text = f"by {author}"
        author_bbox = draw.textbbox((0, 0), author_text, font=author_font)
        author_width = author_bbox[2] - author_bbox[0]
        author_x = (W - author_width) // 2
        draw_text_with_shadow((author_x, author_y), author_text, author_font, "#e0e0e0")
    
    return img

def generate_book_cover(title, plot_description, author="", subtitle="", 
                       num_inference_steps=25, guidance_scale=7.5, add_text=True):
    """Optimized book cover generation"""
    
    global pipeline
    
    if pipeline is None:
        return None, "❌ Model not loaded. Please wait for initialization."
    
    if not title or not plot_description:
        return None, "❌ Please provide both title and plot description."
    
    try:
        # Create optimized prompt
        prompt = f"sks book cover, {plot_description}, professional book cover art, highly detailed, no text"
        
        # Shorter negative prompt for speed
        negative_prompt = "text, words, letters, blurry, low quality, bad art"
        
        print(f"Generating: {prompt[:100]}...")
        
        # Clear GPU cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate image with proper dtype handling
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        with torch.autocast(device, dtype=dtype):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=768,
                width=512,
                generator=torch.Generator(device=device).manual_seed(42)
            )
        
        image = result.images[0]
        
        # Add text overlay if requested
        if add_text:
            image = add_text_to_cover(image, title, author, subtitle)
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return image, f"✅ Book cover generated successfully in {num_inference_steps} steps!"
        
    except Exception as e:
        error_msg = f"❌ Error generating cover: {str(e)}"
        print(error_msg)
        return None, error_msg

def create_gradio_interface():
    """Create the optimized Gradio interface"""
    
    with gr.Blocks(title="📚 Fast AI Book Cover Generator", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 📚 Fast AI Book Cover Generator
        
        Generate professional book covers using AI! This optimized version loads faster and generates covers quicker.
        
        **Quick Start:**
        1. Enter your book title and plot description
        2. Click "Generate Book Cover" (default settings work great!)
        3. Optionally customize settings for different results
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📖 Book Information")
                
                title_input = gr.Textbox(
                    label="Book Title",
                    placeholder="Enter the book title...",
                    value="The Dragon's Quest"
                )
                
                plot_input = gr.Textbox(
                    label="Plot Description", 
                    placeholder="Describe the story, genre, setting, mood...",
                    lines=3,
                    value="epic fantasy adventure with dragons, magical castles, and brave heroes"
                )
                
                with gr.Row():
                    author_input = gr.Textbox(
                        label="Author (Optional)",
                        placeholder="Author's name...",
                        value=""
                    )
                    
                    subtitle_input = gr.Textbox(
                        label="Subtitle (Optional)",
                        placeholder="Book subtitle...", 
                        value=""
                    )
                
                generate_btn = gr.Button("🎨 Generate Book Cover", variant="primary", size="lg")
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    add_text_checkbox = gr.Checkbox(
                        label="Add text overlay to cover",
                        value=True
                    )
                    
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50, 
                        value=25,  # Reduced default for speed
                        step=5,
                        label="Inference Steps (Higher = Better Quality, Slower)"
                    )
                    
                    guidance_slider = gr.Slider(
                        minimum=3.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
                
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ Generated Cover")
                
                output_image = gr.Image(
                    label="Your Book Cover",
                    type="pil",
                    height=600
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to generate...",
                    interactive=False
                )
        
        # Quick examples
        gr.Markdown("### 🚀 Quick Examples (Click to Try)")
        
        examples = [
            ["Space Odyssey", "futuristic space adventure with alien worlds", "A. Sci-Fi", "", 25, 7.5, True],
            ["Dark Forest", "horror mystery in haunted woods", "M. Scary", "", 25, 7.5, True],
            ["Cyber Dreams", "cyberpunk thriller in neon city", "N. Future", "", 25, 7.5, True],
            ["Magic Academy", "young wizard school adventure", "W. Magic", "", 25, 7.5, True]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[title_input, plot_input, author_input, subtitle_input, steps_slider, guidance_slider, add_text_checkbox],
            outputs=[output_image, status_text],
            fn=generate_book_cover,
            cache_examples=False,
            run_on_click=True
        )
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_book_cover,
            inputs=[title_input, plot_input, author_input, subtitle_input, steps_slider, guidance_slider, add_text_checkbox],
            outputs=[output_image, status_text]
        )
        
        gr.Markdown("""
        ---
        ### ⚡ Optimization Features:
        - **Faster Model Loading**: XFormers attention, CPU offload, memory slicing
        - **Quick Generation**: Optimized scheduler (DPM++), reduced default steps
        - **Memory Efficient**: Automatic cleanup, cached fonts, smart batching
        - **Better Performance**: Model warmup, autocast, consistent seeds
        
        ### 💡 Tips:
        - Default settings (25 steps) work great for most covers
        - Increase steps to 35-50 only if you need higher quality
        - Use specific genre keywords for better results
        """)
    
    return demo

def main():
    """Main function to run the optimized Gradio app"""
    
    print("🚀 Starting Optimized Book Cover Generator...")
    
    # Load the model with optimizations
    if not load_model():
        print("❌ Failed to load model. Please check your model files.")
        return
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    print("✅ Gradio interface ready!")
    print("🌐 Launching web interface...")
    
    # Launch with optimized settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False,  # Disable debug for better performance
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
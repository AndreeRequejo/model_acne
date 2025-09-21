# predict_gui.py - Interfaz gráfica simple
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
import numpy as np

# Importar módulos locales
from config import TEST_TRANSFORM, CLASS_NAMES
from model import MyNet

class AcneClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Severidad de Acné")
        self.root.geometry("800x600")
        
        self.model = None
        self.device = None
        self.current_image = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Configurar interfaz de usuario"""
        
        # Frame principal
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = tk.Label(main_frame, text="🏥 Clasificador de Severidad de Acné", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Botones
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(0, 20))
        
        self.load_button = tk.Button(button_frame, text="📁 Cargar Imagen", 
                                    command=self.load_image, font=("Arial", 12))
        self.load_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.predict_button = tk.Button(button_frame, text="🔍 Clasificar", 
                                       command=self.predict, font=("Arial", 12),
                                       state=tk.DISABLED)
        self.predict_button.pack(side=tk.LEFT)
        
        # Frame para imagen
        self.image_frame = tk.Frame(main_frame, relief=tk.SUNKEN, bd=2)
        self.image_frame.pack(pady=(0, 20))
        
        self.image_label = tk.Label(self.image_frame, text="No hay imagen cargada", 
                                   width=40, height=15, bg="lightgray")
        self.image_label.pack(padx=10, pady=10)
        
        # Frame para resultados
        self.result_frame = tk.Frame(main_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Etiquetas de resultado
        self.result_text = tk.Text(self.result_frame, height=10, width=60, 
                                  font=("Courier", 10), state=tk.DISABLED)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar para texto
        scrollbar = tk.Scrollbar(self.result_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Listo")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model(self):
        """Cargar modelo de clasificación"""
        try:
            self.status_var.set("Cargando modelo...")
            self.root.update()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            try:
                model = torch.load("model_acne.pt", weights_only=False, map_location=device)
            except:
                model = MyNet()
                model.load_state_dict(torch.load("model_acne.pt", weights_only=True, map_location=device))
            
            model = model.to(device)
            model.eval()
            
            self.model = model
            self.device = device
            self.status_var.set("Modelo cargado exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{str(e)}")
            self.status_var.set("Error cargando modelo")
    
    def load_image(self):
        """Cargar imagen desde archivo"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Cargar y mostrar imagen
                image = Image.open(file_path).convert('RGB')
                self.current_image = image
                
                # Redimensionar para mostrar
                display_image = image.copy()
                display_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                
                # Convertir para tkinter
                photo = ImageTk.PhotoImage(display_image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                # Habilitar botón de predicción
                self.predict_button.config(state=tk.NORMAL)
                self.status_var.set(f"Imagen cargada: {file_path}")
                
                # Limpiar resultados anteriores
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
    
    def predict(self):
        """Realizar predicción"""
        if self.model is None or self.current_image is None:
            messagebox.showwarning("Advertencia", "Modelo o imagen no disponible")
            return
        
        try:
            self.status_var.set("Clasificando...")
            self.root.update()
            
            # Preprocesar imagen
            image_tensor = TEST_TRANSFORM(self.current_image).unsqueeze(0).to(self.device)
            
            # Hacer predicción
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
                probs_array = probabilities.cpu().numpy()[0]
            
            # Mostrar resultados
            self.display_results(predicted_class, confidence, probs_array)
            self.status_var.set("Clasificación completada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la predicción:\n{str(e)}")
            self.status_var.set("Error en predicción")
    
    def display_results(self, predicted_class, confidence, probabilities):
        """Mostrar resultados de la clasificación"""
        
        # Información de severidad
        severity_info = {
            0: ("LEVE", "Acné leve con pocos comedones"),
            1: ("MODERADO", "Acné moderado con pápulas"),
            2: ("SEVERO", "Acné severo con pústulas"),
            3: ("MUY SEVERO", "Acné muy severo con nódulos")
        }
        
        severity, description = severity_info.get(predicted_class, ("DESCONOCIDO", ""))
        
        # Formatear texto de resultados
        result_text = f"""
╔══════════════════════════════════════════════════════════╗
║                    RESULTADO DE CLASIFICACIÓN            ║
╠══════════════════════════════════════════════════════════╣

🏷️  SEVERIDAD PREDICHA: {CLASS_NAMES[predicted_class]}
📊 NIVEL: {severity}
🎯 CONFIANZA: {confidence:.4f} ({confidence*100:.2f}%)

📝 DESCRIPCIÓN:
   {description}

📊 PROBABILIDADES POR CLASE:
╠══════════════════════════════════════════════════════════╣
"""
        
        # Agregar probabilidades
        for i, prob in enumerate(probabilities):
            marker = ">>> " if i == predicted_class else "    "
            result_text += f"{marker}{CLASS_NAMES[i]:15}: {prob:.4f} ({prob*100:.2f}%)\n"
        
        result_text += f"""
╠══════════════════════════════════════════════════════════╣
💡 RECOMENDACIONES:
"""
        
        # Recomendaciones según severidad
        recommendations = {
            0: "• Limpieza facial suave diaria\n• Productos no comedogénicos\n• Evitar tocar el rostro",
            1: "• Productos con ácido salicílico\n• Considerar consulta dermatológica\n• Mantener rutina de limpieza",
            2: "• Consulta dermatológica recomendada\n• Posible tratamiento tópico intensivo\n• Evitar auto-medicación",
            3: "• CONSULTA DERMATOLÓGICA URGENTE\n• Tratamiento médico necesario\n• Posibles medicamentos sistémicos"
        }
        
        result_text += f"   {recommendations.get(predicted_class, 'Consultar especialista')}\n"
        
        # Advertencia si confianza es baja
        if confidence < 0.6:
            result_text += f"""
╠══════════════════════════════════════════════════════════╣
⚠️  ADVERTENCIA: CONFIANZA BAJA ({confidence:.2f})
   Se recomienda obtener segunda opinión médica
"""
        
        result_text += """
╚══════════════════════════════════════════════════════════╝

NOTA: Este es un sistema de apoyo diagnóstico.
Siempre consulte con un dermatólogo para diagnóstico definitivo.
"""
        
        # Mostrar en el widget de texto
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state=tk.DISABLED)

def main():
    """Función principal para ejecutar la GUI"""
    root = tk.Tk()
    app = AcneClassifierGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Aplicación cerrada por el usuario")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import tkinter
        from PIL import ImageTk
        main()
    except ImportError as e:
        print(f"❌ Error: Falta dependencia - {e}")
        print("Instala las dependencias necesarias:")
        print("pip install pillow")
        print("Para sistemas Linux: sudo apt-get install python3-tk")
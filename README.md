# Frequency Guard

## AI-Generated Image Detection System

🌐 **Live Demo**: [https://frequency-guard-1jqym7e1k-borkarpr-5920s-projects.vercel.app/](https://frequency-guard-1jqym7e1k-borkarpr-5920s-projects.vercel.app/)

Frequency Guard is an advanced deepfake detection tool that analyzes images using frequency-domain analysis, statistical fingerprinting, and multi-resolution feature extraction to identify AI-generated content. The system employs a sophisticated two-stage training pipeline that combines forensic analysis with weak signal learning.

### Key Features

- 🔍 **Multi-Resolution Frequency Analysis**: Contourlet Transform, Multi-Resolution FFT, Phase Randomness Detection
- 📊 **Statistical Fingerprinting**: Fractal Dimension, GLCM Features, Texture Analysis
- 🎯 **Model-Specific Artifact Detection**: Identifies patterns from Stable Diffusion, DALL-E, Midjourney, and other generators
- 🧠 **Weak Signal Learning**: Two-stage training pipeline with forensic analysis and optional user labeling
- 📈 **Training Dashboard**: Monitor model performance, calibration metrics, and per-model accuracy
- 🎨 **Interactive Visualizations**: Frequency spectrum, wavelet analysis, and embedding space visualization
- ⚡ **Real-Time Analysis**: Fast inference with confidence scoring

### Technology Stack

- **Frontend**: React 18 + TypeScript + Vite
- **UI Components**: Radix UI + shadcn/ui + Tailwind CSS
- **State Management**: TanStack Query (React Query)
- **Routing**: React Router v6
- **Charts**: Recharts
- **Form Handling**: React Hook Form + Zod validation

---

## 🚀 Getting Started (Download as ZIP)

### Prerequisites

You need to have **Node.js** (version 18 or higher) installed on your system.

- **Check if Node.js is installed**: Open a terminal/command prompt and run:
  ```bash
  node --version
  ```
  If you see a version number like `v18.x.x` or higher, you're good to go!

- **Don't have Node.js?** Download and install it from:
  - [https://nodejs.org/](https://nodejs.org/) (recommended: LTS version)
  - Or use [nvm](https://github.com/nvm-sh/nvm) for easier version management

### Installation Steps

1. **Download the Project**
   - Click the green "Code" button on GitHub
   - Select "Download ZIP"
   - Extract the ZIP file to your desired location (e.g., `D:\My Projects\frequency-guard`)

2. **Open Terminal in Project Directory**
   - **Windows**: Navigate to the extracted folder, then type `cmd` in the address bar and press Enter
   - **Mac/Linux**: Open Terminal and use `cd` to navigate to the folder:
     ```bash
     cd path/to/frequency-guard
     ```

3. **Install Dependencies**
   ```bash
   npm install
   ```
   This will download all required packages (~200MB). It may take 2-5 minutes.

4. **Start the Development Server**
   ```bash
   npm run dev
   ```
   
5. **Open in Browser**
   - The app will automatically open, or visit: **http://localhost:5173**
   - You should see the Frequency Guard homepage!

### Quick Test

Once the app is running:
1. Go to the main page
2. Upload an image (drag & drop or click to browse)
3. Wait for the analysis to complete
4. View the detection results, confidence score, and forensic analysis

### Available Scripts

- `npm run dev` - Start development server (with hot reload)
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint to check code quality

---

## 📚 Project Structure

```
frequency-guard/
├── src/
│   ├── components/         # React components
│   │   ├── training/       # Training system components
│   │   └── ui/            # Reusable UI components (shadcn/ui)
│   ├── pages/             # Main page components
│   │   ├── Index.tsx      # Home page (detection interface)
│   │   ├── Training.tsx   # Training dashboard
│   │   └── NotFound.tsx   # 404 page
│   ├── lib/
│   │   ├── services/      # Business logic & API calls
│   │   └── types/         # TypeScript type definitions
│   └── hooks/             # Custom React hooks
├── public/                # Static assets
├── package.json           # Node.js dependencies
├── vite.config.ts         # Vite configuration
├── tailwind.config.ts     # Tailwind CSS configuration
└── tsconfig.json          # TypeScript configuration
```

---

## 🎓 Using the Training Section

Navigate to the Training page by clicking "Training" in the header or visiting `http://localhost:5173/training`.

### Training Intake
- Upload images for training
- Automatic forensic analysis (Stage 1) extracts frequency features
- Optional weak labeling (Stage 2) allows you to provide feedback
- Submit labeled data to improve the model

### Dashboard
- Monitor overall accuracy (96.4% baseline)
- View per-model performance metrics
- Check calibration error and unknown AI detection
- Track LOMO cross-validation results

### Model Management
- Add custom AI models to the detection system
- Manage model families (Diffusion, GAN, Transformer-based)
- Configure model-specific detection parameters

### Embedding Visualizer
- View high-dimensional embeddings in 2D/3D space
- Interactive t-SNE/UMAP visualization
- Cluster analysis and anomaly detection

For detailed training documentation, see [TRAINING_SYSTEM.md](TRAINING_SYSTEM.md).

---

## 🚢 Deployment Options

### Option 1: Deploy to Vercel (Recommended)

Vercel is optimized for React/Vite applications and provides zero-configuration deployment.

1. **Push to GitHub** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy to Vercel**:
   - Visit [https://vercel.com](https://vercel.com)
   - Click "Import Project"
   - Connect your GitHub account
   - Select the `frequency-guard` repository
   - Click "Deploy" (Vercel auto-detects Vite configuration)

3. **Your app will be live** at: `https://your-project-name.vercel.app`

The project includes a `vercel.json` configuration file for optimal deployment settings.

### Option 2: Deploy to Netlify

1. **Push to GitHub** (same as above)
2. Visit [https://netlify.com](https://netlify.com)
3. Click "Add new site" → "Import an existing project"
4. Connect to GitHub and select your repository
5. Build settings (auto-detected):
   - Build command: `npm run build`
   - Publish directory: `dist`
6. Click "Deploy site"

### Build for Production (Manual Deployment)

To build the project for manual deployment to any static hosting service:

```bash
npm run build
```

This creates a `dist/` folder with optimized production files. Upload the contents to:
- GitHub Pages
- AWS S3 + CloudFront
- Azure Static Web Apps
- Any static hosting provider

---

## 🐛 Troubleshooting

### Port Already in Use
If port 5173 is already occupied:
```bash
# Windows
netstat -ano | findstr :5173
taskkill /PID <PID_NUMBER> /F

# Mac/Linux
lsof -ti:5173 | xargs kill -9
```

Or run on a different port:
```bash
npm run dev -- --port 3000
```

### Dependencies Installation Fails
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Module Not Found Errors
Make sure you're in the correct directory and all dependencies are installed:
```bash
npm install
```

### Build Errors
Check Node.js version (must be ≥18):
```bash
node --version
```

---

## 📖 Additional Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and technical design
- **[TRAINING_SYSTEM.md](TRAINING_SYSTEM.md)** - Detailed training pipeline documentation
- **[QUICK_START.md](QUICK_START.md)** - 5-minute quick start guide
- **[VERCEL_DEPLOY.md](VERCEL_DEPLOY.md)** - Vercel deployment instructions

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📝 License

This project is open source and available for educational and research purposes.

---

## 🙋‍♂️ Support

If you encounter any issues:
1. Check the [Troubleshooting](#-troubleshooting) section above
2. Review the documentation files in the repository
3. Check that all dependencies are correctly installed
4. Ensure you're using Node.js version 18 or higher

---

**Built with ❤️ using React, TypeScript, and modern web technologies**

Before deploying, test the Streamlit app locally:

```bash
# Install Python dependencies
pip install -r requirements-python.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

---
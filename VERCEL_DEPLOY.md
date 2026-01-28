# Vercel Deployment Guide - Frequency Guard

## 🚀 Quick Deploy

### One-Command Deployment
```bash
npm run deploy
```

### Step-by-Step Deployment

1. **Login to Vercel** (first time only)
   ```bash
   npx vercel login
   ```
   - Visit the authentication URL shown in terminal
   - Authorize with GitHub/Email
   - Press ENTER after authorization

2. **Deploy to Production**
   ```bash
   npx vercel --prod
   ```
   
   Or use the npm script:
   ```bash
   npm run deploy
   ```

3. **Deploy Preview (for testing)**
   ```bash
   npm run deploy:preview
   ```

## ✅ What's Configured

- ✅ **Vercel Configuration** - Optimized `vercel.json` with SPA routing
- ✅ **Security Headers** - X-Frame-Options, Content-Security, XSS Protection
- ✅ **Cache Control** - Assets cached for 1 year, HTML fresh
- ✅ **Build Optimization** - Code splitting for faster loads
- ✅ **Deployment Scripts** - Quick deploy commands in package.json
- ✅ **Ignore Rules** - Excludes Python files and unnecessary assets

## 🔧 Vercel Settings

All settings are pre-configured in `vercel.json`:
- **Framework**: Vite (auto-detected)
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Node Version**: Auto (uses latest LTS)

## 🌐 After Deployment

You'll receive a permanent URL like:
```
https://frequency-guard.vercel.app
```

Or with your custom domain:
```
https://frequency-guard-[random].vercel.app
```

## 🔄 Continuous Deployment

To enable automatic deployments on every git push:

1. Import your project in Vercel Dashboard
2. Connect your GitHub repository
3. Vercel will auto-deploy on every push to main branch

## 📝 Environment Variables

If you need environment variables:
```bash
vercel env add VITE_API_URL production
```

Or add them in Vercel Dashboard under Project Settings > Environment Variables

## 🛠️ Troubleshooting

### Build Fails
```bash
npm run build
```
Fix any errors shown, then deploy again.

### Need to Update Deployment
```bash
npm run deploy
```
This redeploys with latest changes.

### Preview Before Production
```bash
npm run deploy:preview
```
Test your changes on a preview URL first.

## 📦 What Gets Deployed

Only these files are deployed (see `.vercelignore`):
- `dist/` folder (build output)
- `vercel.json` (configuration)
- `package.json` (for build info)

Python files, notebooks, and dev files are excluded for faster deploys.

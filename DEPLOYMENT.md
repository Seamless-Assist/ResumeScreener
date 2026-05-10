# ResumeScreener Deployment Guide

## Overview
This guide covers deploying the ResumeScreener application to a Digital Ocean virtual server.

## Prerequisites
- Digital Ocean account
- Domain name (optional but recommended for SSL)
- SSH access to your server

## Server Requirements
- Ubuntu 22.04 LTS or similar
- At least 1GB RAM (2GB recommended)
- Python 3.12+

## Quick Deployment Options

### Option 1: Using Coolify (Recommended)
If you're using Coolify on Digital Ocean:

1. **Environment Variables**: Set these in your Coolify environment:
   ```
   OPENAI_API_KEY=your_key
   MANATAL_API_KEY=your_key
   MANATAL_BASE_URL=your_url
   SA_WEB_SESSION_SECRET=your_random_secret
   PORT=5050
   ```

2. **Deploy**: The `nixpacks.toml` and `Dockerfile` are already configured for production.

### Option 2: Manual Docker Deployment

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd ResumeScreener
   cp .env.example .env
   # Edit .env with your actual values
   ```

2. **Build and run**:
   ```bash
   docker build -t resumescreener .
   docker run -d -p 5050:5050 --env-file .env resumescreener
   ```

### Option 3: Direct Server Deployment

1. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip nginx
   ```

2. **Setup application**:
   ```bash
   git clone <your-repo>
   cd ResumeScreener
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your values
   ```

3. **Create systemd service**:
   ```bash
   sudo nano /etc/systemd/system/resumescreener.service
   ```

   Add this content:
   ```
   [Unit]
   Description=ResumeScreener Flask App
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/path/to/ResumeScreener
   Environment=PATH=/path/to/ResumeScreener/venv/bin
   ExecStart=/path/to/ResumeScreener/venv/bin/gunicorn --bind 127.0.0.1:5050 --workers 4 --threads 2 wsgi:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

4. **Start service**:
   ```bash
   sudo systemctl enable resumescreener
   sudo systemctl start resumescreener
   ```

## Nginx Configuration (Recommended)

Create `/etc/nginx/sites-available/resumescreener`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static files (if any)
    location /static {
        alias /path/to/ResumeScreener/web/static;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/resumescreener /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## SSL with Let's Encrypt (Optional but Recommended)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Security Considerations

1. **Firewall**: Configure UFW to only allow necessary ports
2. **Environment Variables**: Never commit secrets to version control
3. **Updates**: Keep dependencies updated
4. **Monitoring**: Set up log monitoring and alerts
5. **Backups**: Regular backups of the `cache/` and `results/` directories

## Troubleshooting

- **Port issues**: Ensure port 5050 is not blocked by firewall
- **Permission errors**: Check file permissions for cache/results directories
- **Memory issues**: Monitor RAM usage, consider increasing server size if needed
- **API limits**: Monitor OpenAI and Manatal API usage and rate limits

## Health Check

Test your deployment:
```bash
curl http://your-server-ip:5050/api/roles
```

Should return a JSON array of roles.
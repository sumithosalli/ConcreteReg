from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import pickle
import numpy as np


app = FastAPI()
# Set up templates directory
templates = Jinja2Templates(directory="ConcreteReg/templates")
# Optional: mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="ConcreteReg/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the Templates/index.html template."""
    return templates.TemplateResponse('index.html', {'request': request})


@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    """Render the Templates/contact.html template."""
    return templates.TemplateResponse('contact.html', {'request': request})


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """Render the Templates/about.html template."""
    return templates.TemplateResponse('about.html', {'request': request})


@app.get("/services", response_class=HTMLResponse)
async def services(request: Request):
    """Render the Templates/services.html template."""
    return templates.TemplateResponse('services.html', {'request': request})


@app.get("/documentation", response_class=HTMLResponse)
async def docs(request: Request):
    """Render the Templates/docs.html template."""
    return templates.TemplateResponse('docs.html', {'request': request})


@app.get("/blog", response_class=HTMLResponse)
async def blog(request: Request):
    """Render the Templates/blog.html template."""
    return templates.TemplateResponse('blog.html', {'request': request})


# Load the model
with open('concrete_strength_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  cement: float = Form(...),
                  blast_furnace_slag: float = Form(...),
                  fly_ash: float = Form(...),
                  water: float = Form(...),
                  superplasticizer: float = Form(...),
                  coarse_aggregate: float = Form(...),
                  age: int = Form(...)):
    cement_to_water = cement / (water + 1e-9)
    cement_times_age = cement * age
    features = np.array([[cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate,
                          age, cement_to_water, cement_times_age]])
    prediction = model.predict(features)[0]
    return templates.TemplateResponse("result.html", {"request": request, "prediction": f"{prediction:.2f}"})


if __name__ == '__main__':
    # Run with uvicorn for local development
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)

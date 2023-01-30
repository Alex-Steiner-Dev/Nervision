import replicate

model = replicate.models.get("stability-ai/stable-diffusion")
version = model.versions.get("f178fa7a1ae43a9a9af01b833b9d2ecf97b1bcb0acfd2dc5dd04895e042863f1")

def getImage(prompt):
    inputs = {
        'prompt': prompt,
        'width': 768,
        'height': 768,
        'prompt_strength': 0.2,

        'num_outputs': 1,

        'num_inference_steps': 50,
        'guidance_scale': 7.5,
        'scheduler': "DPMSolverMultistep",

    }

    output = version.predict(**inputs)
    return output

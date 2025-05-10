import json
from myshapes import Sphere, Triangle, Plane
import numpy as np
import matplotlib.pyplot as plt

scene_fn = "scene_5.json"
res = 256

#### Scene Loader

def loadScene(scene_fn):

	with open(scene_fn) as f:
		data = json.load(f)

	spheres = []

	for sphere in data["Spheres"]:
		spheres.append(
			Sphere(sphere["Center"], sphere["Radius"], 
		 	sphere["Mdiff"], sphere["Mspec"], sphere["Mgls"], sphere["Refl"],
		 	sphere["Kd"], sphere["Ks"], sphere["Ka"]))
		
	triangles = []

	for triangle in data["Triangles"]:
		triangles.append(
			Triangle(triangle["A"], triangle["B"], triangle["C"],
			triangle["Mdiff"], triangle["Mspec"], triangle["Mgls"], triangle["Refl"],
			triangle["Kd"], triangle["Ks"], triangle["Ka"]))
	
	planes = []

	for plane in data["Planes"]:
		planes.append(
			Plane(plane["Normal"], plane["Distance"],
			plane["Mdiff"], plane["Mspec"], plane["Mgls"], plane["Refl"],
			plane["Kd"], plane["Ks"], plane["Ka"]))
	
	objects = spheres + triangles + planes

	camera = {
		"LookAt": np.array(data["Camera"]["LookAt"],),
		"LookFrom": np.array(data["Camera"]["LookFrom"]),
		"Up": np.array(data["Camera"]["Up"]),
		"FieldOfView": data["Camera"]["FieldOfView"]
	}

	light = {
		"DirectionToLight": np.array(data["Light"]["DirectionToLight"]),
		"LightColor": np.array(data["Light"]["LightColor"]),
		"AmbientLight": np.array(data["Light"]["AmbientLight"]),
		"BackgroundColor": np.array(data["Light"]["BackgroundColor"]),
	}

	return camera, light, objects

### Ray Tracer

camera, light, objects = loadScene(scene_fn)

image = np.zeros((res,res,3), dtype=np.float32)

# YOUR CODE HERE
"""
Step 1: Defining the Viewing Rays
"""

# Gram-Schmidt Orthologonlization
e3 = (camera["LookAt"] - camera["LookFrom"])/np.linalg.norm(camera["LookAt"] - camera["LookFrom"])
e1 = np.cross(e3, camera["Up"])/np.linalg.norm(np.cross(e3, camera["Up"]))
e2 = np.cross(e1,e3)/np.linalg.norm(np.cross(e1,e3))

# Determining window dimensions
d = np.linalg.norm(camera["LookAt"] - camera["LookFrom"])

fov = np.deg2rad(camera["FieldOfView"])
umax = d*np.tan(fov/2)
vmax = d*np.tan(fov/2)

umin = -umax
vmin = -vmax

# Determining s coord for each pixel
du = (umax - umin)/(res + 1)
dv = (vmax - vmin)/(res + 1)

s_coords = []

for i in range(-128, 128):
	for j in range(-128, 128):
		s = camera["LookAt"] + du*(j+0.5)*e1 + dv*(i+0.5)*e2
		s_coords.append(s)

# Calculate ray direction
r0 = camera["LookFrom"]
rd = []

for s in s_coords:
	dir = (s - r0)/np.linalg.norm(s - r0)
	rd.append(dir)

"""
Step 2: Cast Rays
AND
Step 3: Add Lighting and Shading
"""
# Function to return closest object and its t_min value
def closestObject(ray_o, ray_d, curr_obj=None):
	t_min = -1
	closest_obj = None
	for ob in objects:
		# print(ob.getDiffuse())
		if ob == curr_obj:
			continue

		t_val = ob.intersect(ray_o, ray_d)
		# print(t_val)

		if t_val > 0.000001 and (t_val < t_min or t_min == -1):
			t_min = t_val
			closest_obj = ob

	# print(closest_obj.getDiffuse())
	return t_min, closest_obj

def computeColor(r0, ray_dir, bounce):
	# Get object original ray bounce might hit
	t, obj = closestObject(r0, ray_dir)

	if t <= 0: # If it hits nothing, return background color
		return light["BackgroundColor"]
	
	else: # Hits object

		# Calculate new intersection point of object hit
		p = r0 + (ray_dir*t)

		# Get normal of object
		nhat = 0
		try:
			nhat = obj.getNormal()
		except:
			nhat = obj.getNormal(p)

		# Compute vhat, lhat, and rhat
		vhat = (ray_direction/np.linalg.norm(ray_direction))
		lhat = light["DirectionToLight"]/np.linalg.norm(light["DirectionToLight"])
		rhat = 2*np.sum(-vhat*nhat)*nhat - -vhat
		rhat = rhat/np.linalg.norm(rhat)

		# Determine if ray bounces again and hits another object
		bounce_t, bounce_obj = closestObject(t, rhat)
		if bounce_t > 0 and bounce < 5: # Continue the ray bounce
			bounce += 1
			return computeColor(p, rhat, bounce)
		
		else:
			# Get diffuse, specular, reflectance, and gloss of object
			mdiff = obj.getDiffuse()
			mspec = obj.getSpecular()
			mgls = obj.getGloss()

			# Get kd, ks, ka, and kr of object
			kd = obj.getKd()
			ks = obj.getKs()
			ka = obj.getKa()

			# Get light color and ambience of the light source
			s = light["LightColor"]
			samb = light["AmbientLight"]

			# Calculate final pixel color
			camb = mdiff * samb	# Getting ambient color

			# Determine if object is behind another object relative to light source
			shadow_t, shadow_obj = closestObject(p, lhat, obj)
			if shadow_t > 0: # Give shadow color
				return ka*camb
			
			else:
				"""
				Recompute rhat again to include lhat

				This was something interesting I needed to do for my raytracer. Not sure why, but everything works still so I made do with it
				"""
				rhat = 2*np.sum(lhat*nhat)*nhat - lhat
				rhat = rhat/np.linalg.norm(rhat)

				c = 0	# Initializing final color c
				cdiff_dot = np.sum(nhat*lhat)	# Comput dot product in finding diffuse color

				# Comput c depending on if diffuse color dot product is positive or negative
				if cdiff_dot > 0:	# Dot product is positive

					cdiff = (s*mdiff)*cdiff_dot
					cspec = (s*mspec)*np.sum(vhat*rhat)**mgls

					c = (kd*cdiff) + (ks*cspec) + (ka*camb) #+ (kr*crefl)
				else:	# Dot product is negative
					c = (ka*camb) #+ (kr*crefl)

				# Set color
				return c

# Loop to cast each ray
index = 0
for i in range(res):
	for j in range(res):

		# Get current ray and find closest object to intersection
		ray_direction = rd[index]
		t, obj = closestObject(r0, ray_direction)

		# Assign color to pixel based on t value
		if t <= 0:	# Pixel is given background color
			image[i,j] = light["BackgroundColor"]
			
		else:	# Perform color calculations and assignment

			# Calculate point of intersection p
			p = r0 + (ray_direction*t)

			# Calculate nhat for all objects
			nhat = 0
			try:
				nhat = obj.getNormal()
			except:
				nhat = obj.getNormal(p)

			# Compute vhat, lhat, and rhat
			vhat = (ray_direction/np.linalg.norm(ray_direction))
			lhat = light["DirectionToLight"]/np.linalg.norm(light["DirectionToLight"])

			# Get diffuse, specular, reflectance, and gloss of object
			mdiff = obj.getDiffuse()
			mspec = obj.getSpecular()
			mamb = obj.getRefl()
			mgls = obj.getGloss()

			# Get kd, ks, ka, and kr of object
			kd = obj.getKd()
			ks = obj.getKs()
			ka = obj.getKa()
			kr = obj.getRefl()

			# Determine crefl value
			crefl = 0
			if kr > 0: # Object has reflective value
				rhat = 2*np.sum(-vhat*nhat)*nhat - -vhat
				rhat = rhat/np.linalg.norm(rhat)

				crefl = computeColor(p, rhat, 1) # Get color for crefl
				
			# Get light color and ambience of the light source
			s = light["LightColor"]
			samb = light["AmbientLight"]

			# Calculate final pixel color
			camb = mdiff * samb	# Getting ambient color

			# Determine if object is behind another object relative to light source
			shadow_t, shadow_obj = closestObject(p, lhat, obj)
			if shadow_t > 0: # Give shadow color
				image[i,j] = ka*camb
			
			else:
				rhat = 2*np.sum(lhat*nhat)*nhat - lhat
				rhat = rhat/np.linalg.norm(rhat)

				c = 0	# Initializing final color c
				cdiff_dot = np.sum(nhat*lhat)	# Comput dot product in finding diffuse color

				# Comput c depending on if diffuse color dot product is positive or negative
				if cdiff_dot > 0:	# Dot product is positive

					cdiff = (s*mdiff)*cdiff_dot
					cspec = (s*mspec)*np.sum(vhat*rhat)**mgls

					c = (kd*cdiff) + (ks*cspec) + (ka*camb) + (kr*crefl)
				else:	# Dot product is negative
					c = (ka*camb) + (kr*crefl)

				# Set color
				image[i,j] = c

		index += 1

# Flipping the image
image = np.flipud(image)

### Save and Display Output
plt.imsave("output.png", image)
plt.imshow(image);plt.show()
import sys, os
sys.path.append(os.path.join(os.path.abspath(os.getcwd())))
import BlenderToolbox.BlenderToolBox as bt
import bpy
import time
cwd = os.getcwd()

if not os.path.exists('Renderings'):
    os.makedirs('Renderings')
    
parameters = {"bimba": ((0.58312, 0, 0.99799), (45.8, -6.74, 83.1), 1),
       	      "birdcage": ((0, 0, 0.98), (85.1, 16.2, -198), 0.76),
       	      "bitore": ((0.67525, 0, 0.76669), (0, 0, -159), 0.648),
       	      "buckminsterfullerene": ((-0.20291, 0, 0.87277), (90, 0, 0), 0.808),
       	      "bunny": ((0.44641, 0, 0.55216), (83.6, -1.35, 78.2), 0.557),
              "chair": ((0.44641, 0, 0.065789), (90, 0, 61.9), 0.039),
       	      "dino": ((0, -0.25205, 0.86107), (82.3, 0, 70.1), 0.009),
       	      "dragon": ((0,0,0.48775), (93.6,0,83.4), 0.006),
       	      "fertility": ((0,0,0.85242), (109,0.761,-71.2), 0.75),
       	      "guitar": ((0.58312, 0, 0.99799), (45.8, -6.74, 83.1), 1),
       	      "hand": ((0.57859, 0, 0.63136), (90, 0, 284), 0.666),
       	      "hand2": ((0,0,0.706), (45.9,-7.89,82.9), 0.86),
       	      "helice": ((0.269,0,0.706), (-90,0,0), 0.35),
       	      "hilbert": ((0,0,0.706), (0,0,42), 0.705),
       	      "lamp": ((0, 0, 0.025907), (77, 12.1, -228), 0.01),
       	      "metatron": ((0, 0, 0.60441), (90, 0, 19), 0.632),
       	      "pillowbox": ((0.63455, 0, 0.80642),  (86.5, 7.72, -23.3), 0.432),
       	      "protein": ((0,0,0.915), (0,0,0), 0.814),
       	      "spot": ((-0.02,0.93,0.02), (85,0,-37), 1.614),
       	      "zilla": ((0.42443, 0, 0.63789), (129, -22, -243), 0.9)}

def make_tubes(curves, objects, context, obj, bevel_depth=0.01, resolution=1):

    curve_name = 'TubesCurve'

    mesh = obj.data

    # if exists, pick up else generate a new one
    cu = curves.get(curve_name, curves.new(name=curve_name, type='CURVE'))
    cu.dimensions = '3D'
    cu.fill_mode = 'FULL'
    cu.bevel_depth = bevel_depth
    cu.bevel_resolution = resolution
    cu_obj = objects.get(curve_name, objects.new(curve_name, cu))

    # break down existing splines entirely.
    if cu.splines:
        cu.splines.clear()

    # and rebuild
    verts = mesh.vertices

    for e in mesh.edges:
        idx_v1, idx_v2 = e.vertices
        v0, v1 = verts[idx_v1].co, verts[idx_v2].co

        vc0 = v0.copy()
        vc1 = v1.copy()

        vc0.rotate(obj.rotation_euler)
        vc0[0] = vc0[0]*obj.scale[0]
        vc0[1] = vc0[1]*obj.scale[1]
        vc0[2] = vc0[2]*obj.scale[2]
        vc0 += obj.location

        vc1.rotate(obj.rotation_euler)
        vc1[0] = vc1[0]*obj.scale[0]
        vc1[1] = vc1[1]*obj.scale[1]
        vc1[2] = vc1[2]*obj.scale[2]
        vc1 += obj.location

        full_flat = [vc0[0], vc0[1], vc0[2], 0.0, vc1[0], vc1[1], vc1[2], 0.0]

        # each spline has a default first coordinate but we need two.
        segment = cu.splines.new('POLY')
        segment.points.add(1)
        segment.points.foreach_set('co', full_flat)

    if not curve_name in context.collection.objects:
        context.collection.objects.link(cu_obj)  

    # Create a material
    mat = bpy.data.materials.new("Rededge")
    
    # Activate its nodes
    mat.use_nodes = True
    
    # Get the principled BSDF (created by default)
    principled = mat.node_tree.nodes['Principled BSDF']
    
    # Assign the color
    principled.inputs['Base Color'].default_value = (1,0,0,1)

    cu_obj.data.materials.append(mat)
    print("Edges built.")

def make_spheres(curves, objects, context, obj, scale=0.02):
    # Create a material
    mat = bpy.data.materials.new("Redvertex")
    
    # Activate its nodes
    mat.use_nodes = True
    
    # Get the principled BSDF (created by default)
    principled = mat.node_tree.nodes['Principled BSDF']
    
    # Assign the color
    principled.inputs['Base Color'].default_value = (1,0,0,1)
    
    for v in obj.data.vertices:
        vc = v.co.copy()

        vc.rotate(obj.rotation_euler)
        vc[0] = vc[0]*obj.scale[0]
        vc[1] = vc[1]*obj.scale[1]
        vc[2] = vc[2]*obj.scale[2]
        vc += obj.location
        bpy.ops.mesh.primitive_ico_sphere_add(radius=scale, location=vc) 
        sphere = bpy.context.object
        sphere.color = (1.0,0,0,1.0)
        sphere.data.materials.append(mat)

        for f in sphere.data.polygons:
            f.use_smooth = True 

    print("Sphere vertices built.")
    
    

def render(name):
    t = time.time()
    
    imgRes_x =  1080 # recommend > 1080 
    imgRes_y =  1080 # recommend > 1080 
    numSamples = 100 # recommend > 200
    location = parameters[name][0] # (GUI: click mesh > Transform > Location)
    rotation = parameters[name][1] # (GUI: click mesh > Transform > Rotation)
    scale_unif = parameters[name][2]
    scale = (scale_unif, scale_unif, scale_unif) # (GUI: click mesh > Transform > Scale)
    
    outputPath = os.path.join(cwd, './Renderings/{}.png'.format(name))
    
    ## initialize blender
    exposure = 1.5 
    use_GPU = True
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure, use_GPU)
    
    ## read skeleton
    meshPath = 'Results/cvsk_Sine_{}.obj'.format(name)
    mesh = bt.readMesh(meshPath, location, rotation, scale)
    
    #add color to mesh skeleton
    mat = mesh.active_material
    mat.diffuse_color = [1.0, 0.5, 0.5 ,1.0]
    inputs = mat.node_tree.nodes["Principled BSDF"].inputs
    inputs["Base Color"].default_value =  (0.9,0.1,0.1,1)
    
    
    #create and add color to the skeleton edges and vertices
    make_tubes(bpy.data.curves, bpy.data.objects, bpy.context, mesh)
    make_spheres(bpy.data.curves, bpy.data.objects, bpy.context, mesh)
    
    
    #read corresponding mesh
    meshPath2 = 'Objects/{}.obj'.format(name)
    #location2 = (0, 0, -0.75) # (GUI: click mesh > Transform > Location)
    #rotation2 = (90, 0, 227) # (GUI: click mesh > Transform > Rotation)
    #scale2 = (1,1,1) # (GUI: click mesh > Transform > Scale)
    mesh2 = bt.readMesh(meshPath2, location, rotation, scale)
    
    #add color to transparent mesh
    gray = (0.5, 0.5, 0.5, 1)
    meshColor2 = bt.colorObj(gray)
    alpha = 0.4
    transmission = 1
    bt.setMat_transparent(mesh2, meshColor2, alpha, transmission)
    
    bpy.context.view_layer.update()
    
    ## set shading (uncomment one of them)
    bpy.ops.object.shade_smooth() # Option1: Gouraud shading
    #bpy.ops.object.shade_flat() # Option2: Flat shading
    # bt.edgeNormals(mesh, angle = 10) # Option3: Edge normal shading
    
    ## subdivision
    #bt.subdivision(mesh, level = 1)
        
    
    ## End material
    ###########################################
    
    ## set invisible plane (shadow catcher)
    bt.invisibleGround(shadowBrightness=0.9)
    
    ## set camera 
    ## Option 1: don't change camera setting, change the mesh location above instead
    camLocation = (3, 0, 2)
    lookAtLocation = (0,0,0.5)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
    ## Option 2: if you really want to set camera based on the values in GUI, then
    # camLocation = (3, 0, 2)
    # rotation_euler = (63,0,90)
    # focalLength = 45
    # cam = bt.setCamera_from_UI(camLocation, rotation_euler, focalLength = 35)
    
    ## set light
    ## Option1: Three Point Light System 
    # bt.setLight_threePoints(radius=4, height=10, intensity=1700, softness=6, keyLoc='left')
    ## Option2: simple sun light
    lightAngle = (6, -30, -155) 
    strength = 2
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
    
    ## set ambient light
    bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 
    
    ## set gray shadow to completely white with a threshold (optional but recommended)
    bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')
    
    ## save blender file so that you can adjust parameters in the UI
    bpy.ops.wm.save_mainfile(filepath=os.getcwd() + './Renderings/{}.blend'.format(name))
    
    ## save rendering
    bt.renderImage(outputPath, cam)
    
    print("Total computation time ", '{:.2f}'.format(time.time()-t)," s.")
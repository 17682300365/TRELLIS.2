import aspose.threed as a3d
#
scene = a3d.Scene.from_file("chassis.jt")
scene.save("Output.glb")

# exec('import aspose.threed')
# exec('import aspose.threed.formats')
# print('import aspose m',aspose.threed,aspose.threed.formats)
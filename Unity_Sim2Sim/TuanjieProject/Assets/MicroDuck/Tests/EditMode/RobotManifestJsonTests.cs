using System;
using NUnit.Framework;

namespace AgenticRobot.MicroDuck.Tests
{
    public sealed class RobotManifestJsonTests
    {
        [Test]
        public void ParsesVersionedPhysicsAndJointData()
        {
            RobotManifestData manifest = RobotManifestJson.Parse(ValidManifestJson());

            Assert.That(manifest.schemaVersion, Is.EqualTo(1));
            Assert.That(manifest.variant, Is.EqualTo("legged"));
            Assert.That(manifest.timestep, Is.EqualTo(0.005f).Within(1e-7f));
            Assert.That(manifest.gravity, Is.EqualTo(new[] { 0f, -9.81f, 0f }));
            Assert.That(manifest.bodies[0].mass, Is.EqualTo(0.42f).Within(1e-7f));
            Assert.That(manifest.joints[1].axis, Is.EqualTo(new[] { 0f, -1f, 0f }));
            Assert.That(manifest.servos[0].jointName, Is.EqualTo("hinge"));
            Assert.That(
                manifest.geoms[0].friction,
                Is.EqualTo(new[] { 1f, 0.005f, 0.0001f }));
        }

        [Test]
        public void RejectsUnknownBodyParent()
        {
            string json = ValidManifestJson().Replace(
                "\"parentName\":\"root\"",
                "\"parentName\":\"missing\"");

            ManifestValidationException error = Assert.Throws<ManifestValidationException>(
                () => RobotManifestJson.Parse(json));

            StringAssert.Contains("missing", error.Message);
        }

        [Test]
        public void RejectsNonContiguousServoIndices()
        {
            string json = ValidManifestJson().Replace("\"index\":0", "\"index\":1");

            ManifestValidationException error = Assert.Throws<ManifestValidationException>(
                () => RobotManifestJson.Parse(json));

            StringAssert.Contains("index", error.Message.ToLowerInvariant());
        }

        [Test]
        public void RejectsMalformedGeomFrictionVectors()
        {
            string json = ValidManifestJson().Replace(
                "[1.0,0.005,0.0001]",
                "[1.0,0.005]");

            ManifestValidationException error = Assert.Throws<ManifestValidationException>(
                () => RobotManifestJson.Parse(json));

            StringAssert.Contains("friction", error.Message.ToLowerInvariant());
        }

        private static string ValidManifestJson()
        {
            return @"{
  ""schemaVersion"":1,
  ""source"":{""relativePath"":""scene.xml"",""sha256"":""aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"",""modelName"":""scene""},
  ""variant"":""legged"",
  ""timestep"":0.005,
  ""gravity"":[0.0,-9.81,0.0],
  ""bodies"":[
    {""name"":""root"",""parentName"":null,""localPosition"":[0,0.12,0],""localRotationWxyz"":[1,0,0,0],""mass"":0.42,""centerOfMass"":[0,0,0],""inertiaTensor"":[0.1,0.1,0.1],""inertiaRotationWxyz"":[1,0,0,0]},
    {""name"":""child"",""parentName"":""root"",""localPosition"":[0,-0.1,0],""localRotationWxyz"":[1,0,0,0],""mass"":0.1,""centerOfMass"":[0,0,0],""inertiaTensor"":[0.01,0.01,0.01],""inertiaRotationWxyz"":[1,0,0,0]}
  ],
  ""joints"":[
    {""name"":""root_free"",""bodyName"":""root"",""type"":""free"",""localAnchor"":[0,0,0],""axis"":[0,1,0],""limited"":false,""rangeRad"":[0,0],""damping"":[0,0,0,0,0,0],""frictionLoss"":[0,0,0,0,0,0],""armature"":[0,0,0,0,0,0],""qposAddress"":0,""dofAddress"":0,""passive"":false},
    {""name"":""hinge"",""bodyName"":""child"",""type"":""hinge"",""localAnchor"":[0,0,0],""axis"":[0,-1,0],""limited"":true,""rangeRad"":[-1,1],""damping"":[0.05],""frictionLoss"":[0.004],""armature"":[0.001],""qposAddress"":7,""dofAddress"":6,""passive"":false}
  ],
  ""servos"":[{""name"":""hinge"",""jointName"":""hinge"",""index"":0,""ctrlLimited"":true,""ctrlRange"":[-10,10],""forceLimited"":true,""forceRange"":[-0.96,0.96],""gainPrm"":[0.55],""biasPrm"":[0,-0.55],""gear"":[1]}],
  ""passiveJointNames"":[],
  ""geoms"":[
    {""name"":""foot_collision"",""bodyName"":""child"",""type"":""box"",""localPosition"":[0,0,0],""localRotationWxyz"":[1,0,0,0],""size"":[0.1,0.1,0.1],""meshName"":null,""rgba"":[0.2,0.3,0.4,1],""friction"":[1.0,0.005,0.0001],""contype"":1,""conaffinity"":1,""group"":3,""collidable"":true}
  ],
  ""meshes"":[],""sensors"":[],""keyframes"":[]
}";
        }
    }
}

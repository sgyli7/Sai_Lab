using System;
using System.Collections.Generic;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public sealed class ManifestValidationException : Exception
    {
        public ManifestValidationException(string message) : base(message)
        {
        }
    }

    [Serializable]
    public sealed class RobotManifestData
    {
        public int schemaVersion;
        public ManifestSourceData source;
        public string variant;
        public float timestep;
        public float[] gravity;
        public ManifestBodyData[] bodies;
        public ManifestJointData[] joints;
        public ManifestServoData[] servos;
        public string[] passiveJointNames;
        public ManifestGeomData[] geoms;
        public ManifestMeshData[] meshes;
        public ManifestSensorData[] sensors;
        public ManifestKeyframeData[] keyframes;
    }

    [Serializable]
    public sealed class ManifestSourceData
    {
        public string relativePath;
        public string sha256;
        public string modelName;
    }

    [Serializable]
    public sealed class ManifestBodyData
    {
        public string name;
        public string parentName;
        public float[] localPosition;
        public float[] localRotationWxyz;
        public float mass;
        public float[] centerOfMass;
        public float[] inertiaTensor;
        public float[] inertiaRotationWxyz;
    }

    [Serializable]
    public sealed class ManifestJointData
    {
        public string name;
        public string bodyName;
        public string type;
        public float[] localAnchor;
        public float[] axis;
        public bool limited;
        public float[] rangeRad;
        public float[] damping;
        public float[] frictionLoss;
        public float[] armature;
        public int qposAddress;
        public int dofAddress;
        public bool passive;
    }

    [Serializable]
    public sealed class ManifestServoData
    {
        public string name;
        public string jointName;
        public int index;
        public bool ctrlLimited;
        public float[] ctrlRange;
        public bool forceLimited;
        public float[] forceRange;
        public float[] gainPrm;
        public float[] biasPrm;
        public float[] gear;
    }

    [Serializable]
    public sealed class ManifestGeomData
    {
        public string name;
        public string bodyName;
        public string type;
        public float[] localPosition;
        public float[] localRotationWxyz;
        public float[] size;
        public string meshName;
        public float[] rgba;
        public float[] friction;
        public int contype;
        public int conaffinity;
        public int group;
        public bool collidable;
    }

    [Serializable]
    public sealed class ManifestMeshData
    {
        public string name;
        public string sourceFile;
        public string sourceSha256;
        public string assetFile;
        public string assetSha256;
        public string collisionAssetFile;
        public string collisionAssetSha256;
        public float[] scale;
    }

    [Serializable]
    public sealed class ManifestSensorData
    {
        public string name;
        public string type;
        public int dim;
        public string objectType;
        public string objectName;
    }

    [Serializable]
    public sealed class ManifestKeyframeData
    {
        public string name;
        public float[] qpos;
        public float[] qvel;
        public float[] ctrl;
    }

    public static class RobotManifestJson
    {
        public static RobotManifestData Parse(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                throw new ManifestValidationException("Manifest JSON is empty.");
            }

            RobotManifestData manifest;
            try
            {
                manifest = JsonUtility.FromJson<RobotManifestData>(json);
            }
            catch (ArgumentException error)
            {
                throw new ManifestValidationException($"Manifest JSON is malformed: {error.Message}");
            }

            Validate(manifest);
            return manifest;
        }

        public static void Validate(RobotManifestData manifest)
        {
            Require(manifest != null, "Manifest is null.");
            Require(manifest.schemaVersion == 1, $"Unsupported schemaVersion {manifest.schemaVersion}.");
            Require(manifest.source != null, "Manifest source is missing.");
            Require(!string.IsNullOrWhiteSpace(manifest.source.relativePath), "Source relativePath is missing.");
            Require(IsSha256(manifest.source.sha256), "Source sha256 must contain 64 hexadecimal characters.");
            Require(manifest.variant == "legged" || manifest.variant == "roller", "Variant must be legged or roller.");
            Require(IsFinitePositive(manifest.timestep), "Timestep must be finite and positive.");
            RequireVector(manifest.gravity, 3, "gravity");
            Require(manifest.bodies != null && manifest.bodies.Length > 0, "Manifest has no bodies.");
            Require(manifest.joints != null, "Manifest joints are missing.");
            Require(manifest.servos != null, "Manifest servos are missing.");
            Require(manifest.geoms != null, "Manifest geoms are missing.");
            Require(manifest.meshes != null, "Manifest meshes are missing.");

            var bodyNames = new HashSet<string>(StringComparer.Ordinal);
            int rootCount = 0;
            foreach (ManifestBodyData body in manifest.bodies)
            {
                Require(body != null && !string.IsNullOrWhiteSpace(body.name), "A body name is missing.");
                Require(bodyNames.Add(body.name), $"Duplicate body '{body.name}'.");
                RequireVector(body.localPosition, 3, $"body {body.name} localPosition");
                RequireQuaternion(body.localRotationWxyz, $"body {body.name} localRotationWxyz");
                Require(IsFinitePositive(body.mass), $"Body '{body.name}' mass must be finite and positive.");
                RequireVector(body.centerOfMass, 3, $"body {body.name} centerOfMass");
                RequirePositiveVector(body.inertiaTensor, $"body {body.name} inertiaTensor");
                RequireQuaternion(body.inertiaRotationWxyz, $"body {body.name} inertiaRotationWxyz");
                if (string.IsNullOrEmpty(body.parentName))
                {
                    rootCount++;
                }
            }

            Require(rootCount == 1, $"Manifest must have exactly one root body; found {rootCount}.");
            foreach (ManifestBodyData body in manifest.bodies)
            {
                if (!string.IsNullOrEmpty(body.parentName))
                {
                    Require(bodyNames.Contains(body.parentName), $"Body '{body.name}' references missing parent '{body.parentName}'.");
                    Require(body.parentName != body.name, $"Body '{body.name}' cannot parent itself.");
                }
            }

            var jointNames = new HashSet<string>(StringComparer.Ordinal);
            foreach (ManifestJointData joint in manifest.joints)
            {
                Require(joint != null && !string.IsNullOrWhiteSpace(joint.name), "A joint name is missing.");
                Require(jointNames.Add(joint.name), $"Duplicate joint '{joint.name}'.");
                Require(bodyNames.Contains(joint.bodyName), $"Joint '{joint.name}' references missing body '{joint.bodyName}'.");
                Require(joint.type == "free" || joint.type == "hinge", $"Joint '{joint.name}' has unsupported type '{joint.type}'.");
                RequireVector(joint.localAnchor, 3, $"joint {joint.name} localAnchor");
                RequireVector(joint.axis, 3, $"joint {joint.name} axis");
                RequireVector(joint.rangeRad, 2, $"joint {joint.name} rangeRad");
            }

            foreach (ManifestGeomData geom in manifest.geoms)
            {
                Require(geom != null && !string.IsNullOrWhiteSpace(geom.name), "A geom name is missing.");
                RequireVector(geom.friction, 3, $"geom {geom.name} friction");
                foreach (float coefficient in geom.friction)
                {
                    Require(coefficient >= 0f, $"Geom '{geom.name}' friction values must be non-negative.");
                }
            }

            var servoJoints = new HashSet<string>(StringComparer.Ordinal);
            for (int index = 0; index < manifest.servos.Length; index++)
            {
                ManifestServoData servo = manifest.servos[index];
                Require(servo != null, $"Servo at index {index} is null.");
                Require(servo.index == index, $"Servo index must be contiguous: expected {index}, found {servo.index}.");
                Require(jointNames.Contains(servo.jointName), $"Servo '{servo.name}' references missing joint '{servo.jointName}'.");
                Require(servoJoints.Add(servo.jointName), $"Joint '{servo.jointName}' has more than one servo.");
                RequireVector(servo.ctrlRange, 2, $"servo {servo.name} ctrlRange");
                RequireVector(servo.forceRange, 2, $"servo {servo.name} forceRange");
            }
        }

        private static bool IsSha256(string value)
        {
            if (value == null || value.Length != 64)
            {
                return false;
            }

            foreach (char character in value)
            {
                bool digit = character >= '0' && character <= '9';
                bool lower = character >= 'a' && character <= 'f';
                bool upper = character >= 'A' && character <= 'F';
                if (!digit && !lower && !upper)
                {
                    return false;
                }
            }

            return true;
        }

        private static bool IsFinitePositive(float value)
        {
            return value > 0f && !float.IsNaN(value) && !float.IsInfinity(value);
        }

        private static void RequireVector(float[] values, int expected, string label)
        {
            Require(values != null && values.Length == expected, $"{label} must contain {expected} values.");
            foreach (float value in values)
            {
                Require(!float.IsNaN(value) && !float.IsInfinity(value), $"{label} contains a non-finite value.");
            }
        }

        private static void RequirePositiveVector(float[] values, string label)
        {
            RequireVector(values, 3, label);
            foreach (float value in values)
            {
                Require(value > 0f, $"{label} values must be positive.");
            }
        }

        private static void RequireQuaternion(float[] values, string label)
        {
            RequireVector(values, 4, label);
            float normSquared = 0f;
            foreach (float value in values)
            {
                normSquared += value * value;
            }

            Require(normSquared > 0.5f && normSquared < 1.5f, $"{label} is not a valid unit quaternion.");
        }

        private static void Require(bool condition, string message)
        {
            if (!condition)
            {
                throw new ManifestValidationException(message);
            }
        }
    }
}

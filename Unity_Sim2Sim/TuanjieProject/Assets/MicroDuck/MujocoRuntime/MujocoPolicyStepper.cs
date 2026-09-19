using System;
using System.Runtime.InteropServices;
using Mujoco;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Mujoco
{
    /// <summary>
    /// Connects the exact official MicroDuck MuJoCo state to the shared 61D -> 14D
    /// policy loop. The model and data remain owned by the caller (normally MjScene).
    /// </summary>
    public sealed unsafe class MujocoPolicyStepper : IDisposable
    {
        private static readonly float[] HomePositionRad =
        {
            0f, -0.0873f, -0.4579f, -0.0049f, 0.4530f,
            0.3491f, 0.3491f, 0f, 0f,
            0f, 0.0873f, 0.4579f, 0.0049f, -0.4530f,
        };

        private static readonly string[] PassiveWheelJointNames =
        {
            "passive_LF_wheel",
            "passive_LR_wheel",
            "passive_RF_wheel",
            "passive_RR_wheel",
        };

        private readonly MujocoLib.mjModel_* model;
        private readonly MujocoLib.mjData_* data;
        private readonly RobotVariant robotVariant;
        private readonly int[] jointQposAddresses = new int[PolicyContract.ActionCount];
        private readonly int[] jointQvelAddresses = new int[PolicyContract.ActionCount];
        private readonly float[] jointPosition = new float[PolicyContract.ActionCount];
        private readonly float[] jointVelocity = new float[PolicyContract.ActionCount];
        private readonly float[] targets = new float[PolicyContract.ActionCount];
        private readonly MicroDuckControlLoop controlLoop;
        private readonly int rootQposAddress;
        private readonly int trunkBodyId;
        private readonly int gyroAddress;
        private bool disposed;

        public MujocoPolicyStepper(
            MujocoLib.mjModel_* model,
            MujocoLib.mjData_* data,
            IPolicyRuntime runtime,
            PolicyCommandState commands,
            RobotVariant robotVariant,
            float actionScale,
            MicroDuckControlLoopContinuation continuation = null)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            if (data == null)
            {
                throw new ArgumentNullException(nameof(data));
            }

            if (model->nu != PolicyContract.ActionCount)
            {
                throw new ArgumentException(
                    $"MicroDuck MuJoCo model must expose {PolicyContract.ActionCount} actuators; found {model->nu}.",
                    nameof(model));
            }

            this.model = model;
            this.data = data;
            this.robotVariant = robotVariant;
            ServoNames = new string[PolicyContract.ActionCount];
            for (int actuatorId = 0; actuatorId < PolicyContract.ActionCount; actuatorId++)
            {
                int jointId = model->actuator_trnid[(2 * actuatorId)];
                if (jointId < 0)
                {
                    throw new ArgumentException($"Actuator {actuatorId} has no joint transmission.", nameof(model));
                }

                jointQposAddresses[actuatorId] = model->jnt_qposadr[jointId];
                jointQvelAddresses[actuatorId] = model->jnt_dofadr[jointId];
                ServoNames[actuatorId] = NameFor(MujocoLib.mjtObj.mjOBJ_ACTUATOR, actuatorId);
            }

            int rootJointId = RequiredId(MujocoLib.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint");
            rootQposAddress = model->jnt_qposadr[rootJointId];
            trunkBodyId = RequiredId(MujocoLib.mjtObj.mjOBJ_BODY, "trunk_base");
            int gyroSensorId = RequiredId(MujocoLib.mjtObj.mjOBJ_SENSOR, "imu_ang_vel");
            gyroAddress = model->sensor_adr[gyroSensorId];
            controlLoop = new MicroDuckControlLoop(
                runtime,
                commands,
                HomePositionRad,
                actionScale,
                continuation);
        }

        public string[] ServoNames { get; }
        public string BackendName => controlLoop.BackendName;
        public float[] LastObservation => controlLoop.LastObservation;
        public float[] LastRawAction => controlLoop.LastRawAction;
        public float[] LastCommand => controlLoop.LastCommand;
        public float[] LastTargets => (float[])targets.Clone();

        public MicroDuckControlLoopContinuation CaptureContinuation()
        {
            ThrowIfDisposed();
            return controlLoop.CaptureContinuation();
        }

        public void ResetToHome()
        {
            ThrowIfDisposed();
            controlLoop.Reset();
            MujocoLib.mj_resetData(model, data);
            data->qpos[rootQposAddress] = 0.0;
            data->qpos[rootQposAddress + 1] = 0.0;
            data->qpos[rootQposAddress + 2] =
                robotVariant == RobotVariant.Roller ? 0.1385 : 0.125;
            data->qpos[rootQposAddress + 3] = 1.0;
            data->qpos[rootQposAddress + 4] = 0.0;
            data->qpos[rootQposAddress + 5] = 0.0;
            data->qpos[rootQposAddress + 6] = 0.0;
            for (int index = 0; index < PolicyContract.ActionCount; index++)
            {
                data->qpos[jointQposAddresses[index]] = HomePositionRad[index];
                data->ctrl[index] = HomePositionRad[index];
                targets[index] = HomePositionRad[index];
            }

            if (robotVariant == RobotVariant.Roller)
            {
                foreach (string jointName in PassiveWheelJointNames)
                {
                    int jointId = RequiredId(MujocoLib.mjtObj.mjOBJ_JOINT, jointName);
                    model->dof_frictionloss[model->jnt_dofadr[jointId]] = 0.003;
                }
            }

            MujocoLib.mj_forward(model, data);
        }

        public void ResetToPose(
            Vector3 leggedRootPosition,
            float yawDegrees,
            Vector3 initialLinearVelocityMetersPerSecond)
        {
            ResetToHome();
            Vector3 position = leggedRootPosition;
            if (robotVariant == RobotVariant.Roller)
            {
                position += Vector3.up * (0.1385f - 0.125f);
            }

            MjEngineTool.SetMjVector3(data->qpos + rootQposAddress, position);
            MjEngineTool.SetMjQuaternion(
                data->qpos + rootQposAddress + 3,
                Quaternion.Euler(0f, yawDegrees, 0f));
            int rootJointId = RequiredId(
                MujocoLib.mjtObj.mjOBJ_JOINT,
                "trunk_base_freejoint");
            int rootDofAddress = model->jnt_dofadr[rootJointId];
            MjEngineTool.SetMjVector3(
                data->qvel + rootDofAddress,
                initialLinearVelocityMetersPerSecond);
            MujocoLib.mj_forward(model, data);
        }

        public bool TickPolicy(float nowSeconds, out string error)
        {
            ThrowIfDisposed();
            for (int index = 0; index < PolicyContract.ActionCount; index++)
            {
                jointPosition[index] = (float)data->qpos[jointQposAddresses[index]];
                jointVelocity[index] = (float)data->qvel[jointQvelAddresses[index]];
            }

            var gyro = new Vector3(
                (float)data->sensordata[gyroAddress],
                (float)data->sensordata[gyroAddress + 1],
                (float)data->sensordata[gyroAddress + 2]);
            double* quaternion = data->xquat + (4 * trunkBodyId);
            Vector3 projectedGravity = RotateInverse(
                (float)quaternion[0],
                new Vector3(
                    (float)quaternion[1],
                    (float)quaternion[2],
                    (float)quaternion[3]),
                new Vector3(0f, 0f, -1f));

            bool succeeded = controlLoop.StepMuJoCo(
                gyro,
                projectedGravity,
                jointPosition,
                jointVelocity,
                nowSeconds,
                targets,
                out error);
            for (int index = 0; index < PolicyContract.ActionCount; index++)
            {
                data->ctrl[index] = targets[index];
            }

            return succeeded;
        }

        public void ApplyLastTargets()
        {
            ThrowIfDisposed();
            for (int index = 0; index < PolicyContract.ActionCount; index++)
            {
                data->ctrl[index] = targets[index];
            }
        }

        public void PlaceBallForRole(PolicyRole role)
        {
            ThrowIfDisposed();
            int ballJointId = MujocoLib.mj_name2id(
                model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "ball_free");
            if (ballJointId < 0)
            {
                if (role == PolicyRole.KickLeft || role == PolicyRole.KickRight)
                {
                    throw new InvalidOperationException(
                        "The active MuJoCo model has no ball_free joint required by a kick policy.");
                }

                return;
            }

            int ballQposAddress = model->jnt_qposadr[ballJointId];
            int ballQvelAddress = model->jnt_dofadr[ballJointId];
            if (role != PolicyRole.KickLeft && role != PolicyRole.KickRight)
            {
                // Keep the always-imported kick ball outside the playable area for
                // policies whose official scenario uses scene.xml without a ball.
                data->qpos[ballQposAddress] = 50.0;
                data->qpos[ballQposAddress + 1] = 50.0;
                data->qpos[ballQposAddress + 2] = 0.035;
            }
            else
            {
                double w = data->qpos[rootQposAddress + 3];
                double x = data->qpos[rootQposAddress + 4];
                double y = data->qpos[rootQposAddress + 5];
                double z = data->qpos[rootQposAddress + 6];
                double yaw = Math.Atan2(
                    2.0 * ((w * z) + (x * y)),
                    1.0 - (2.0 * ((y * y) + (z * z))));
                double lateral = role == PolicyRole.KickLeft ? 0.042 : -0.042;
                data->qpos[ballQposAddress] = data->qpos[rootQposAddress]
                    + (Math.Cos(yaw) * 0.09) - (Math.Sin(yaw) * lateral);
                data->qpos[ballQposAddress + 1] = data->qpos[rootQposAddress + 1]
                    + (Math.Sin(yaw) * 0.09) + (Math.Cos(yaw) * lateral);
                data->qpos[ballQposAddress + 2] = 0.035;
            }

            data->qpos[ballQposAddress + 3] = 1.0;
            data->qpos[ballQposAddress + 4] = 0.0;
            data->qpos[ballQposAddress + 5] = 0.0;
            data->qpos[ballQposAddress + 6] = 0.0;
            for (int offset = 0; offset < 6; offset++)
            {
                data->qvel[ballQvelAddress + offset] = 0.0;
            }

            MujocoLib.mj_forward(model, data);
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            controlLoop.Dispose();
            disposed = true;
        }

        private static Vector3 RotateInverse(float w, Vector3 xyz, Vector3 vector)
        {
            Vector3 cross = 2f * Vector3.Cross(xyz, vector);
            return vector - (w * cross) + Vector3.Cross(xyz, cross);
        }

        private int RequiredId(MujocoLib.mjtObj objectType, string objectName)
        {
            int id = MujocoLib.mj_name2id(model, (int)objectType, objectName);
            if (id < 0)
            {
                throw new ArgumentException(
                    $"Required MuJoCo {objectType} '{objectName}' was not found.",
                    nameof(model));
            }

            return id;
        }

        private string NameFor(MujocoLib.mjtObj objectType, int id)
        {
            IntPtr pointer = MujocoLib.mj_id2name(model, (int)objectType, id);
            string value = Marshal.PtrToStringAnsi(pointer);
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new ArgumentException(
                    $"MuJoCo {objectType} {id} has no stable name.",
                    nameof(model));
            }

            return value;
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(MujocoPolicyStepper));
            }
        }
    }
}

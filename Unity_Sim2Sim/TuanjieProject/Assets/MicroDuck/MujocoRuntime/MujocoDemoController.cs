using System;
using Mujoco;
using Unity.Barracuda;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Mujoco
{
    /// <summary>
    /// Playable MicroDuck controller whose physics authority is the official
    /// MuJoCo runtime and whose neural-network authority is Barracuda.
    /// </summary>
    [DefaultExecutionOrder(-1000)]
    public sealed unsafe class MujocoDemoController : MonoBehaviour
    {
        [SerializeField] private MjScene scene;
        [SerializeField] private GameObject leggedModelRoot;
        [SerializeField] private GameObject rollerModelRoot;
        [SerializeField] private PolicyModelBinding[] policies = Array.Empty<PolicyModelBinding>();
        [SerializeField, Range(1, 9)] private int initialPolicySlot = 2;
        [SerializeField, Min(1)] private int physicsStepsPerPolicyStep = 4;

        private readonly PolicyCommandState commands = new PolicyCommandState();
        private MujocoPolicyStepper stepper;
        private IPolicyRuntime pendingRuntime;
        private RobotVariant pendingVariant;
        private PolicyEntry activeEntry;
        private int physicsStep;
        private bool subscribed;
        private bool resetAfterSceneInitialization;
        private ModelRestPose leggedRestPose;
        private ModelRestPose rollerRestPose;
        private Vector3 resetPositionMeters = new Vector3(0f, 0.125f, 0f);
        private float resetYawDegrees;
        private Vector3 resetInitialVelocityMetersPerSecond;

        public int ActivePolicySlot { get; private set; }
        public string ActivePolicyName { get; private set; } = "not loaded";
        public string BackendName => stepper == null
            ? "not loaded"
            : $"MuJoCo 3.12 + {stepper.BackendName}";
        public string Fault { get; private set; } = string.Empty;
        public bool IsHealthy => stepper != null && string.IsNullOrEmpty(Fault);
        public int PolicyTicks { get; private set; }
        public float[] LastObservation => stepper?.LastObservation ?? Array.Empty<float>();
        public float[] LastRawAction => stepper?.LastRawAction ?? Array.Empty<float>();
        public float[] LastCommand => stepper?.LastCommand ?? Array.Empty<float>();
        public float[] LastTargets => stepper?.LastTargets ?? Array.Empty<float>();
        public Vector3 RootPositionMeters
        {
            get
            {
                if (scene == null || scene.Model == null || scene.Data == null)
                {
                    return Vector3.zero;
                }

                int jointId = MujocoLib.mj_name2id(
                    scene.Model,
                    (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                    "trunk_base_freejoint");
                if (jointId < 0)
                {
                    return Vector3.zero;
                }

                return MjEngineTool.UnityVector3(
                    scene.Data->qpos + scene.Model->jnt_qposadr[jointId]);
            }
        }
        public float TrunkUpright
        {
            get
            {
                if (scene == null || scene.Model == null || scene.Data == null)
                {
                    return 0f;
                }

                int bodyId = MujocoLib.mj_name2id(
                    scene.Model,
                    (int)MujocoLib.mjtObj.mjOBJ_BODY,
                    "trunk_base");
                return bodyId < 0 ? 0f : (float)scene.Data->xmat[(9 * bodyId) + 8];
            }
        }
        public GameObject ActiveModelRoot => activeEntry.RobotVariant == RobotVariant.Roller
            ? rollerModelRoot
            : leggedModelRoot;
        public Vector3 ResetPositionMeters => resetPositionMeters;
        public float ResetYawDegrees => resetYawDegrees;

        public void Configure(
            MjScene configuredScene,
            GameObject leggedRoot,
            GameObject rollerRoot,
            PolicyModelBinding[] modelBindings,
            int initialSlot = 2)
        {
            scene = configuredScene ?? throw new ArgumentNullException(nameof(configuredScene));
            leggedModelRoot = leggedRoot ?? throw new ArgumentNullException(nameof(leggedRoot));
            rollerModelRoot = rollerRoot ?? throw new ArgumentNullException(nameof(rollerRoot));
            policies = modelBindings ?? throw new ArgumentNullException(nameof(modelBindings));
            initialPolicySlot = initialSlot;
            if (Application.isPlaying)
            {
                CaptureModelRestPoses();
            }
        }

        public bool SwitchPolicy(int slot)
        {
            if (stepper != null && ActivePolicySlot > 0)
            {
                PolicyEntry requested;
                try
                {
                    requested = PolicyCatalog.GetBySlot(slot);
                }
                catch (ArgumentOutOfRangeException error)
                {
                    Fault = error.Message;
                    return false;
                }

                if (requested.RobotVariant == activeEntry.RobotVariant)
                {
                    return HotSwapPolicy(slot);
                }
            }

            return SelectPolicy(slot);
        }

        public bool SelectPolicy(int slot)
        {
            if (!TryResolvePolicy(slot, out PolicyEntry entry, out PolicyModelBinding binding))
            {
                return false;
            }

            GameObject requestedRoot = entry.RobotVariant == RobotVariant.Roller
                ? rollerModelRoot
                : leggedModelRoot;
            if (requestedRoot == null)
            {
                Fault = $"No {entry.RobotVariant} MuJoCo model root is configured.";
                return false;
            }

            bool changesRobotVariant = ActivePolicySlot > 0
                && entry.RobotVariant != activeEntry.RobotVariant;
            if (changesRobotVariant)
            {
                ModelRestPose restPose = entry.RobotVariant == RobotVariant.Roller
                    ? rollerRestPose
                    : leggedRestPose;
                restPose?.Restore();
            }

            IPolicyRuntime runtime;
            try
            {
                runtime = new BarracudaPolicyRuntime(binding.model);
            }
            catch (Exception error)
            {
                Fault = $"Could not load {entry.FileName}: {error.Message}";
                return false;
            }

            stepper?.Dispose();
            stepper = null;
            pendingRuntime?.Dispose();
            pendingRuntime = runtime;
            pendingVariant = entry.RobotVariant;
            commands.SelectSlot(slot);
            activeEntry = entry;
            ActivePolicySlot = slot;
            ActivePolicyName = entry.FileName;
            PolicyTicks = 0;
            physicsStep = 0;
            Fault = string.Empty;

            if (leggedModelRoot != null)
            {
                leggedModelRoot.SetActive(entry.RobotVariant == RobotVariant.Legged);
            }

            if (rollerModelRoot != null)
            {
                rollerModelRoot.SetActive(entry.RobotVariant == RobotVariant.Roller);
            }

            if (scene != null && scene.Model != null && ModelMatches(entry.RobotVariant, scene.Model))
            {
                BindPendingRuntime(scene.Model, scene.Data);
            }

            return string.IsNullOrEmpty(Fault);
        }

        public bool HotSwapPolicy(int slot)
        {
            if (stepper == null || scene == null || scene.Model == null || ActivePolicySlot <= 0)
            {
                Fault = "MuJoCo controller has no initialized policy model to hot-swap.";
                return false;
            }

            if (!TryResolvePolicy(slot, out PolicyEntry entry, out PolicyModelBinding binding))
            {
                return false;
            }

            if (entry.RobotVariant != activeEntry.RobotVariant)
            {
                Fault = $"Cannot hot-swap from {activeEntry.RobotVariant} to {entry.RobotVariant}; "
                    + "use SelectPolicy when changing MuJoCo models.";
                return false;
            }

            IPolicyRuntime runtime = null;
            MujocoPolicyStepper replacement = null;
            try
            {
                MicroDuckControlLoopContinuation continuation = stepper.CaptureContinuation();
                runtime = new BarracudaPolicyRuntime(binding.model);
                replacement = new MujocoPolicyStepper(
                    scene.Model,
                    scene.Data,
                    runtime,
                    commands,
                    entry.RobotVariant,
                    entry.ActionScale,
                    continuation);
                runtime = null;
            }
            catch (Exception error)
            {
                replacement?.Dispose();
                runtime?.Dispose();
                Fault = $"Could not hot-swap to {entry.FileName}: {error.Message}";
                return false;
            }

            stepper.Dispose();
            stepper = replacement;
            commands.SelectSlot(slot);
            activeEntry = entry;
            ActivePolicySlot = slot;
            ActivePolicyName = entry.FileName;
            physicsStep = 0;
            Fault = string.Empty;
            try
            {
                stepper.PlaceBallForRole(entry.Role);
            }
            catch (Exception error)
            {
                Fault = error.Message;
                return false;
            }

            return true;
        }

        public void SetTwist(float forward, float left, float yawRate)
        {
            commands.SetTwist(forward, left, yawRate);
        }

        public void SetHead(float neckPitch, float headPitch, float headYaw, float headRoll)
        {
            commands.SetHead(neckPitch, headPitch, headYaw, headRoll);
        }

        public void SetBody(float height, float roll, float pitch)
        {
            commands.SetBody(height, roll, pitch);
        }

        public void TriggerSkill()
        {
            commands.TriggerSkill(CurrentMuJoCoTime());
        }

        public void TriggerSkill(float nowSeconds)
        {
            commands.TriggerSkill(nowSeconds);
        }

        public void ResetActiveRobot()
        {
            if (stepper == null)
            {
                Fault = "MuJoCo controller has not initialized its active model.";
                return;
            }

            try
            {
                commands.Reset();
                ResetBoundNativeState();
                resetAfterSceneInitialization = false;
                physicsStep = 0;
                PolicyTicks = 0;
                Fault = string.Empty;
            }
            catch (Exception error)
            {
                Fault = $"MuJoCo reset failed: {error.Message}";
            }
        }

        public bool ResetActiveRobotAt(
            Vector3 leggedRootPosition,
            float yawDegrees,
            Vector3 initialVelocityMetersPerSecond)
        {
            resetPositionMeters = leggedRootPosition;
            resetYawDegrees = yawDegrees;
            resetInitialVelocityMetersPerSecond = initialVelocityMetersPerSecond;
            if (stepper == null)
            {
                return false;
            }

            ResetActiveRobot();
            return IsHealthy;
        }

        private void Awake()
        {
            CaptureModelRestPoses();
            Subscribe();
        }

        private void Start()
        {
            SelectPolicy(initialPolicySlot);
        }

        private void OnDestroy()
        {
            if (subscribed && scene != null)
            {
                scene.preDestroyEvent -= OnScenePreDestroy;
                scene.postInitEvent -= OnSceneInitialized;
                scene.ctrlCallback -= OnControlCallback;
            }

            stepper?.Dispose();
            pendingRuntime?.Dispose();
        }

        private void Subscribe()
        {
            if (subscribed)
            {
                return;
            }

            if (scene == null)
            {
                scene = MjScene.Instance;
            }

            scene.preDestroyEvent += OnScenePreDestroy;
            scene.postInitEvent += OnSceneInitialized;
            scene.ctrlCallback += OnControlCallback;
            subscribed = true;
        }

        private void OnScenePreDestroy(object sender, MjStepArgs args)
        {
            // MjScene can be destroyed before this sibling component during a
            // scene unload. Release every pointer-backed object while MuJoCo's
            // model/data are still alive so no later callback can dereference
            // stale native memory.
            stepper?.Dispose();
            stepper = null;
            resetAfterSceneInitialization = false;

            // The official plug-in can legitimately coalesce component changes
            // into more than one recreation when a model hierarchy is switched
            // outside our LateUpdate keyboard path. The first recreation consumes
            // pendingRuntime; prepare a fresh runtime for every later recreation
            // so the active policy cannot be left detached from the new model.
            if (pendingRuntime == null && ActivePolicySlot > 0)
            {
                PolicyModelBinding binding = FindBinding(ActivePolicySlot);
                if (binding == null || binding.model == null)
                {
                    Fault = $"Cannot restore {ActivePolicyName} after MuJoCo recreation: "
                        + "its Barracuda model binding is missing.";
                    return;
                }

                try
                {
                    pendingRuntime = new BarracudaPolicyRuntime(binding.model);
                    pendingVariant = activeEntry.RobotVariant;
                }
                catch (Exception error)
                {
                    Fault = $"Cannot restore {ActivePolicyName} after MuJoCo recreation: "
                        + error.Message;
                }
            }
        }

        private void OnSceneInitialized(object sender, MjStepArgs args)
        {
            if (pendingRuntime != null)
            {
                BindPendingRuntime(args.model, args.data);
            }
        }

        private void OnControlCallback(object sender, MjStepArgs args)
        {
            if (stepper == null)
            {
                return;
            }

            if (resetAfterSceneInitialization)
            {
                // MjScene.postInitEvent fires from inside RecreateScene before the
                // official plug-in rehydrates cached joints. Reset on the first
                // control callback as well, after RecreateScene has fully returned.
                try
                {
                    ResetBoundNativeState();
                    resetAfterSceneInitialization = false;
                    physicsStep = 0;
                    PolicyTicks = 0;
                }
                catch (Exception error)
                {
                    Fault = $"Could not reset {ActivePolicyName} after MuJoCo rebuild: "
                        + error.Message;
                    return;
                }
            }

            if (physicsStep++ % physicsStepsPerPolicyStep == 0)
            {
                bool success = stepper.TickPolicy((float)args.data->time, out string error);
                Fault = error;
                PolicyTicks++;
                if (!success)
                {
                    return;
                }
            }

            // MjActuator.OnSyncState mirrors its serialized Control after each step,
            // so the authoritative policy target is deliberately restored every step.
            stepper.ApplyLastTargets();
        }

        private void BindPendingRuntime(
            MujocoLib.mjModel_* model,
            MujocoLib.mjData_* data)
        {
            IPolicyRuntime runtime = pendingRuntime;
            if (runtime == null)
            {
                return;
            }

            pendingRuntime = null;
            try
            {
                stepper = new MujocoPolicyStepper(
                    model,
                    data,
                    runtime,
                    commands,
                    pendingVariant,
                    activeEntry.ActionScale);
                runtime = null;
                resetAfterSceneInitialization = true;
                physicsStep = 0;
                Fault = string.Empty;
            }
            catch (Exception error)
            {
                stepper?.Dispose();
                stepper = null;
                resetAfterSceneInitialization = false;
                runtime?.Dispose();
                Fault = $"Could not bind {ActivePolicyName} to MuJoCo: {error.Message}";
            }
        }

        private bool TryResolvePolicy(
            int slot,
            out PolicyEntry entry,
            out PolicyModelBinding binding)
        {
            try
            {
                entry = PolicyCatalog.GetBySlot(slot);
            }
            catch (ArgumentOutOfRangeException error)
            {
                entry = default;
                binding = null;
                Fault = error.Message;
                return false;
            }

            binding = FindBinding(slot);
            if (binding == null || binding.model == null)
            {
                Fault = $"No Barracuda model is bound to policy slot {slot} ({entry.FileName}).";
                return false;
            }

            return true;
        }

        private PolicyModelBinding FindBinding(int slot)
        {
            if (policies == null)
            {
                return null;
            }

            foreach (PolicyModelBinding binding in policies)
            {
                if (binding != null && binding.slot == slot)
                {
                    return binding;
                }
            }

            return null;
        }

        private static bool ModelMatches(
            RobotVariant variant,
            MujocoLib.mjModel_* model)
        {
            bool hasPassiveWheel = MujocoLib.mj_name2id(
                model,
                (int)MujocoLib.mjtObj.mjOBJ_JOINT,
                "passive_LF_wheel") >= 0;
            return variant == RobotVariant.Roller ? hasPassiveWheel : !hasPassiveWheel;
        }

        private float CurrentMuJoCoTime()
        {
            return scene != null && scene.Data != null ? (float)scene.Data->time : Time.fixedTime;
        }

        private void ResetBoundNativeState()
        {
            Vector3 initialVelocity = activeEntry.RobotVariant == RobotVariant.Roller
                ? resetInitialVelocityMetersPerSecond
                : Vector3.zero;
            stepper.ResetToPose(resetPositionMeters, resetYawDegrees, initialVelocity);
            stepper.PlaceBallForRole(activeEntry.Role);
            scene.SyncUnityToMjState();
        }

        private void CaptureModelRestPoses()
        {
            if (leggedModelRoot != null && leggedRestPose == null)
            {
                leggedRestPose = new ModelRestPose(leggedModelRoot);
            }

            if (rollerModelRoot != null && rollerRestPose == null)
            {
                rollerRestPose = new ModelRestPose(rollerModelRoot);
            }
        }

        /// <summary>
        /// MjScene mutates imported body transforms and joint readouts while it
        /// runs. An inactive variant therefore cannot be recompiled from its
        /// last rendered pose; capture and restore the imported prefab pose.
        /// </summary>
        private sealed class ModelRestPose
        {
            private readonly TransformState[] transforms;
            private readonly HingeState[] hinges;

            public ModelRestPose(GameObject root)
            {
                Transform[] sourceTransforms = root.GetComponentsInChildren<Transform>(true);
                transforms = new TransformState[sourceTransforms.Length];
                for (int index = 0; index < sourceTransforms.Length; index++)
                {
                    transforms[index] = new TransformState(sourceTransforms[index]);
                }

                MjHingeJoint[] sourceHinges = root.GetComponentsInChildren<MjHingeJoint>(true);
                hinges = new HingeState[sourceHinges.Length];
                for (int index = 0; index < sourceHinges.Length; index++)
                {
                    hinges[index] = new HingeState(sourceHinges[index]);
                }
            }

            public void Restore()
            {
                foreach (TransformState state in transforms)
                {
                    state.Restore();
                }

                foreach (HingeState state in hinges)
                {
                    state.Restore();
                }
            }

            private readonly struct TransformState
            {
                private readonly Transform target;
                private readonly Vector3 localPosition;
                private readonly Quaternion localRotation;
                private readonly Vector3 localScale;

                public TransformState(Transform target)
                {
                    this.target = target;
                    localPosition = target.localPosition;
                    localRotation = target.localRotation;
                    localScale = target.localScale;
                }

                public void Restore()
                {
                    if (target == null)
                    {
                        return;
                    }

                    target.localPosition = localPosition;
                    target.localRotation = localRotation;
                    target.localScale = localScale;
                }
            }

            private readonly struct HingeState
            {
                private readonly MjHingeJoint target;
                private readonly float configuration;
                private readonly float velocity;
                private readonly double rawConfiguration;

                public HingeState(MjHingeJoint target)
                {
                    this.target = target;
                    configuration = target.Configuration;
                    velocity = target.Velocity;
                    rawConfiguration = target.RawConfiguration;
                }

                public void Restore()
                {
                    if (target == null)
                    {
                        return;
                    }

                    target.Configuration = configuration;
                    target.Velocity = velocity;
                    target.RawConfiguration = rawConfiguration;
                }
            }
        }
    }
}

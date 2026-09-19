using System;
using Unity.Barracuda;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    [Serializable]
    public sealed class PolicyModelBinding
    {
        [Range(1, 9)] public int slot;
        public NNModel model;
    }

    public sealed class MicroDuckDemoController : MonoBehaviour
    {
        private static readonly float[] HomePositionRad =
        {
            0f, -0.0873f, -0.4579f, -0.0049f, 0.4530f,
            0.3491f, 0.3491f, 0f, 0f,
            0f, 0.0873f, 0.4579f, 0.0049f, -0.4530f,
        };

        [SerializeField] private MicroDuckRig leggedRig;
        [SerializeField] private MicroDuckRig rollerRig;
        [SerializeField] private MicroDuckSkillBall skillBall;
        [SerializeField] private PolicyModelBinding[] policies = Array.Empty<PolicyModelBinding>();
        [SerializeField, Range(1, 9)] private int initialPolicySlot = 2;
        [SerializeField, Min(1)] private int physicsStepsPerPolicyStep = 4;

        private readonly PolicyCommandState commands = new PolicyCommandState();
        private readonly float[] jointPosition = new float[PolicyContract.ActionCount];
        private readonly float[] jointVelocity = new float[PolicyContract.ActionCount];
        private readonly float[] targets = new float[PolicyContract.ActionCount];
        private MicroDuckControlLoop controlLoop;
        private MicroDuckRig activeRig;
        private int physicsStep;

        public int ActivePolicySlot { get; private set; }
        public string ActivePolicyName { get; private set; } = "not loaded";
        public string BackendName => controlLoop?.BackendName ?? "not loaded";
        public string Fault { get; private set; } = string.Empty;
        public bool IsHealthy => string.IsNullOrEmpty(Fault) && controlLoop != null;
        public int PolicyTicks { get; private set; }
        public float[] LastObservation => controlLoop?.LastObservation ?? Array.Empty<float>();
        public float[] LastRawAction => controlLoop?.LastRawAction ?? Array.Empty<float>();
        public float[] LastCommand => controlLoop?.LastCommand ?? Array.Empty<float>();
        public float[] LastTargets => (float[])targets.Clone();
        public float[] LastJointPositionRad => (float[])jointPosition.Clone();
        public float[] LastJointVelocityRadPerSecond => (float[])jointVelocity.Clone();
        public MicroDuckRig ActiveRig => activeRig;
        public MicroDuckSkillBall SkillBall => skillBall;

        public void Configure(
            MicroDuckRig legged,
            MicroDuckRig roller,
            PolicyModelBinding[] modelBindings,
            int initialSlot = 2)
        {
            Configure(legged, roller, modelBindings, null, initialSlot);
        }

        public void Configure(
            MicroDuckRig legged,
            MicroDuckRig roller,
            PolicyModelBinding[] modelBindings,
            MicroDuckSkillBall configuredSkillBall,
            int initialSlot = 2)
        {
            leggedRig = legged ?? throw new ArgumentNullException(nameof(legged));
            rollerRig = roller ?? throw new ArgumentNullException(nameof(roller));
            policies = modelBindings ?? throw new ArgumentNullException(nameof(modelBindings));
            skillBall = configuredSkillBall;
            initialPolicySlot = initialSlot;
            skillBall?.Hide();
        }

        public bool SelectPolicy(int slot)
        {
            PolicyEntry entry;
            try
            {
                entry = PolicyCatalog.GetBySlot(slot);
            }
            catch (ArgumentOutOfRangeException error)
            {
                Fault = error.Message;
                return false;
            }

            PolicyModelBinding binding = FindBinding(slot);
            if (binding == null || binding.model == null)
            {
                Fault = $"No Barracuda model is bound to policy slot {slot} ({entry.FileName}).";
                return false;
            }

            MicroDuckRig requestedRig = entry.RobotVariant == RobotVariant.Roller ? rollerRig : leggedRig;
            if (requestedRig == null)
            {
                Fault = $"No {entry.RobotVariant} rig is configured.";
                return false;
            }

            MicroDuckControlLoop nextLoop;
            try
            {
                nextLoop = new MicroDuckControlLoop(
                    new BarracudaPolicyRuntime(binding.model), commands, HomePositionRad, entry.ActionScale);
            }
            catch (Exception error)
            {
                Fault = $"Could not load {entry.FileName}: {error.Message}";
                return false;
            }

            controlLoop?.Dispose();
            controlLoop = nextLoop;
            commands.SelectSlot(slot);
            ActivePolicySlot = slot;
            ActivePolicyName = entry.FileName;
            activeRig = requestedRig;
            if (leggedRig != null)
            {
                leggedRig.gameObject.SetActive(entry.RobotVariant == RobotVariant.Legged);
            }

            if (rollerRig != null)
            {
                rollerRig.gameObject.SetActive(entry.RobotVariant == RobotVariant.Roller);
            }

            physicsStep = 0;
            PolicyTicks = 0;
            Fault = string.Empty;
            activeRig.ResetPose(HomePositionRad);
            ConfigureSkillBall(entry);
            return true;
        }

        public bool HotSwapPolicy(int slot)
        {
            if (controlLoop == null || activeRig == null || ActivePolicySlot <= 0)
            {
                Fault = "Controller has no active policy and rig to hot-swap.";
                return false;
            }

            PolicyEntry entry;
            try
            {
                entry = PolicyCatalog.GetBySlot(slot);
            }
            catch (ArgumentOutOfRangeException error)
            {
                Fault = error.Message;
                return false;
            }

            PolicyEntry currentEntry = PolicyCatalog.GetBySlot(ActivePolicySlot);
            if (entry.RobotVariant != currentEntry.RobotVariant)
            {
                Fault = $"Cannot hot-swap from {currentEntry.RobotVariant} to {entry.RobotVariant}; "
                    + "use SelectPolicy when changing robot variants.";
                return false;
            }

            PolicyModelBinding binding = FindBinding(slot);
            if (binding == null || binding.model == null)
            {
                Fault = $"No Barracuda model is bound to policy slot {slot} ({entry.FileName}).";
                return false;
            }

            MicroDuckControlLoop nextLoop;
            try
            {
                MicroDuckControlLoopContinuation continuation = controlLoop.CaptureContinuation();
                nextLoop = new MicroDuckControlLoop(
                    new BarracudaPolicyRuntime(binding.model),
                    commands,
                    HomePositionRad,
                    entry.ActionScale,
                    continuation);
            }
            catch (Exception error)
            {
                Fault = $"Could not load {entry.FileName}: {error.Message}";
                return false;
            }

            controlLoop.Dispose();
            controlLoop = nextLoop;
            commands.SelectSlot(slot);
            ActivePolicySlot = slot;
            ActivePolicyName = entry.FileName;
            Fault = string.Empty;
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
            commands.TriggerSkill();
        }

        public void TriggerSkill(float nowSeconds)
        {
            commands.TriggerSkill(nowSeconds);
        }

        public void ResetActiveRig()
        {
            activeRig?.ResetPose(HomePositionRad);
            if (activeRig != null && ActivePolicySlot > 0)
            {
                ConfigureSkillBall(PolicyCatalog.GetBySlot(ActivePolicySlot));
            }
        }

        public bool TickOnce(float nowSeconds)
        {
            if (controlLoop == null || activeRig == null)
            {
                Fault = "Controller has no active policy and rig.";
                return false;
            }

            try
            {
                skillBall?.SetObservationTime(nowSeconds);
                activeRig.ReadPolicyState(
                    jointPosition,
                    jointVelocity,
                    out Vector3 localAngularVelocity,
                    out Vector3 localProjectedGravity);
                bool success = controlLoop.Step(
                    localAngularVelocity,
                    localProjectedGravity,
                    jointPosition,
                    jointVelocity,
                    nowSeconds,
                    targets,
                    out string error);
                activeRig.ApplyTargets(targets);
                Fault = error;
                PolicyTicks++;
                return success;
            }
            catch (Exception error)
            {
                Fault = $"Control tick failed: {error.Message}";
                return false;
            }
        }

        private void Start()
        {
            SelectPolicy(initialPolicySlot);
        }

        private void FixedUpdate()
        {
            if (physicsStep++ % physicsStepsPerPolicyStep == 0)
            {
                TickOnce(Time.fixedTime);
            }
        }

        private void OnDestroy()
        {
            controlLoop?.Dispose();
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

        private void ConfigureSkillBall(PolicyEntry entry)
        {
            if (skillBall == null)
            {
                return;
            }

            if (entry.Role == PolicyRole.KickLeft || entry.Role == PolicyRole.KickRight)
            {
                skillBall.ResetForKick(activeRig.RootBody.transform, entry.Role);
            }
            else
            {
                skillBall.Hide();
            }
        }
    }
}

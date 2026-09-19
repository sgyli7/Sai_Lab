using System.Linq;
using Mujoco;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Mujoco
{
    public enum MujocoCameraMode
    {
        Follow,
        FreeFly,
    }

    public enum MujocoCameraPreset
    {
        Side,
        Rear,
        Top,
        Showcase,
    }

    /// <summary>
    /// Hybrid operator camera for the native MicroDuck scene. Follow mode keeps
    /// the active trunk framed; FreeFly mode temporarily owns navigation input.
    /// </summary>
    [DefaultExecutionOrder(2000)]
    [RequireComponent(typeof(Camera))]
    public sealed class MujocoCameraRig : MonoBehaviour
    {
        [SerializeField] private MujocoDemoController controller;
        [SerializeField] private MujocoCameraMode mode = MujocoCameraMode.Follow;
        [SerializeField] private MujocoCameraPreset preset = MujocoCameraPreset.Side;
        [SerializeField] private float yawDegrees;
        [SerializeField] private float pitchDegrees = 12f;
        [SerializeField] private float distanceMeters = 1.55f;
        [SerializeField] private float lookHeight = 0.08f;
        [SerializeField] private float followSharpness = 12f;
        [SerializeField] private float orbitSensitivity = 0.22f;
        [SerializeField] private float freeLookSensitivity = 0.16f;
        [SerializeField] private float freeSpeedMetersPerSecond = 3f;

        private GameObject activeRoot;
        private Transform target;
        private bool forceSnap = true;

        public MujocoCameraMode Mode => mode;
        public MujocoCameraPreset Preset => preset;
        public bool OwnsNavigationInput => mode == MujocoCameraMode.FreeFly;
        public float PitchDegrees => pitchDegrees;
        public float DistanceMeters => distanceMeters;

        public void Configure(MujocoDemoController value)
        {
            controller = value;
            ApplyPresetState(preset);
            forceSnap = true;
        }

        public void ToggleMode()
        {
            mode = mode == MujocoCameraMode.Follow
                ? MujocoCameraMode.FreeFly
                : MujocoCameraMode.Follow;
            if (mode == MujocoCameraMode.Follow)
            {
                forceSnap = true;
            }
        }

        public void FocusOnRobot()
        {
            mode = MujocoCameraMode.Follow;
            ApplyPresetState(preset);
            forceSnap = true;
        }

        public MujocoCameraPreset CyclePreset()
        {
            int count = System.Enum.GetValues(typeof(MujocoCameraPreset)).Length;
            preset = (MujocoCameraPreset)(((int)preset + 1) % count);
            mode = MujocoCameraMode.Follow;
            ApplyPresetState(preset);
            forceSnap = true;
            return preset;
        }

        public void ApplyOrbitInput(Vector2 delta)
        {
            yawDegrees = Mathf.Repeat(yawDegrees + (delta.x * orbitSensitivity), 360f);
            pitchDegrees = Mathf.Clamp(
                pitchDegrees - (delta.y * orbitSensitivity),
                8f,
                82f);
        }

        public void ApplyZoomInput(float delta)
        {
            distanceMeters = Mathf.Clamp(distanceMeters - (delta * 0.22f), 0.65f, 7f);
        }

        private void LateUpdate()
        {
            if (Input.GetKeyDown(KeyCode.Tab))
            {
                ToggleMode();
            }
            if (Input.GetKeyDown(KeyCode.C))
            {
                CyclePreset();
            }
            if (Input.GetKeyDown(KeyCode.F))
            {
                FocusOnRobot();
            }

            if (mode == MujocoCameraMode.FreeFly)
            {
                UpdateFreeFly();
                return;
            }

            if (Input.GetMouseButton(0))
            {
                ApplyOrbitInput(new Vector2(Input.GetAxisRaw("Mouse X"), Input.GetAxisRaw("Mouse Y")));
            }
            ApplyZoomInput(Input.mouseScrollDelta.y);
            UpdateFollow();
        }

        private void UpdateFollow()
        {
            if (!TryResolveTarget(out bool targetChanged))
            {
                return;
            }

            Vector3 focus = target.position + (Vector3.up * lookHeight);
            Quaternion orbit = Quaternion.Euler(pitchDegrees, yawDegrees, 0f);
            Vector3 desiredPosition = focus + (orbit * (Vector3.back * distanceMeters));
            if (targetChanged || forceSnap || followSharpness <= 0f)
            {
                transform.position = desiredPosition;
                forceSnap = false;
            }
            else
            {
                float blend = 1f - Mathf.Exp(-followSharpness * Time.unscaledDeltaTime);
                transform.position = Vector3.Lerp(transform.position, desiredPosition, blend);
            }

            Vector3 lookDirection = focus - transform.position;
            if (lookDirection.sqrMagnitude > 1e-8f)
            {
                transform.rotation = Quaternion.LookRotation(lookDirection, Vector3.up);
            }
        }

        private void UpdateFreeFly()
        {
            if (Input.GetMouseButton(1))
            {
                Vector3 euler = transform.eulerAngles;
                float pitch = euler.x > 180f ? euler.x - 360f : euler.x;
                pitch = Mathf.Clamp(
                    pitch - (Input.GetAxisRaw("Mouse Y") * freeLookSensitivity),
                    -85f,
                    85f);
                float yaw = euler.y + (Input.GetAxisRaw("Mouse X") * freeLookSensitivity);
                transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
            }

            float wheel = Input.mouseScrollDelta.y;
            if (Mathf.Abs(wheel) > 0.001f)
            {
                freeSpeedMetersPerSecond = Mathf.Clamp(
                    freeSpeedMetersPerSecond * Mathf.Pow(1.2f, wheel),
                    0.35f,
                    14f);
            }

            ApplyFreeFlyTranslation(
                Axis(KeyCode.S, KeyCode.W),
                Axis(KeyCode.A, KeyCode.D),
                Axis(KeyCode.Q, KeyCode.E),
                Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift));
        }

        public void ApplyFreeFlyTranslation(float forward, float right, float up, bool boost)
        {
            Vector3 flatForward = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
            Vector3 flatRight = Vector3.ProjectOnPlane(transform.right, Vector3.up).normalized;
            Vector3 movement = (flatForward * forward)
                + (flatRight * right)
                + (Vector3.up * up);
            float speedBoost = boost ? 2.5f : 1f;
            if (movement.sqrMagnitude > 1f)
            {
                movement.Normalize();
            }

            transform.position += movement
                * freeSpeedMetersPerSecond
                * speedBoost
                * Time.unscaledDeltaTime;
        }

        private bool TryResolveTarget(out bool targetChanged)
        {
            targetChanged = false;
            if (controller == null)
            {
                return false;
            }

            GameObject requestedRoot = controller.ActiveModelRoot;
            if (requestedRoot == null)
            {
                return false;
            }
            if (requestedRoot == activeRoot && target != null)
            {
                return true;
            }

            MjBody trunk = requestedRoot
                .GetComponentsInChildren<MjBody>(includeInactive: false)
                .FirstOrDefault(body => body.name == "trunk_base");
            if (trunk == null)
            {
                return false;
            }

            activeRoot = requestedRoot;
            target = trunk.transform;
            targetChanged = true;
            return true;
        }

        private void ApplyPresetState(MujocoCameraPreset value)
        {
            switch (value)
            {
                case MujocoCameraPreset.Side:
                    yawDegrees = 0f;
                    pitchDegrees = 12f;
                    distanceMeters = 1.55f;
                    break;
                case MujocoCameraPreset.Rear:
                    yawDegrees = 90f;
                    pitchDegrees = 13f;
                    distanceMeters = 1.65f;
                    break;
                case MujocoCameraPreset.Top:
                    yawDegrees = 0f;
                    pitchDegrees = 78f;
                    distanceMeters = 2.2f;
                    break;
                case MujocoCameraPreset.Showcase:
                    yawDegrees = 35f;
                    pitchDegrees = 30f;
                    distanceMeters = 5.8f;
                    break;
            }
        }

        private static float Axis(KeyCode negative, KeyCode positive)
        {
            return (Input.GetKey(positive) ? 1f : 0f)
                - (Input.GetKey(negative) ? 1f : 0f);
        }
    }
}

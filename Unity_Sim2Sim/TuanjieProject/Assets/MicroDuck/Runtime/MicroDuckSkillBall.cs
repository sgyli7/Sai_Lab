using System;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Rigidbody), typeof(SphereCollider))]
    public sealed class MicroDuckSkillBall : MonoBehaviour
    {
        public const float RadiusMeters = 0.035f;
        public const float MassKilograms = 0.015f;
        public const float MomentOfInertiaKilogramMetersSquared = 1.225e-5f;
        public const float ForwardOffsetMeters = 0.09f;
        public const float LateralOffsetMeters = 0.042f;

        [SerializeField] private Rigidbody body;
        [SerializeField] private SphereCollider sphereCollider;
        [SerializeField] private Collider floorCollider;

        private float observationTimeSeconds;

        public Rigidbody Body
        {
            get
            {
                EnsureComponents();
                return body;
            }
        }

        public SphereCollider Collider
        {
            get
            {
                EnsureComponents();
                return sphereCollider;
            }
        }

        public Vector3 Position => Body.position;
        public Vector3 Velocity => Body.velocity;
        public Vector3 AngularVelocity => Body.angularVelocity;
        public bool HasNonFloorContact { get; private set; }
        public string FirstNonFloorContactColliderName { get; private set; } = string.Empty;
        public float FirstNonFloorContactTimeSeconds { get; private set; } = -1f;
        public Vector3 FirstNonFloorContactPoint { get; private set; }

        public void Configure(
            Rigidbody configuredBody,
            SphereCollider configuredCollider,
            Collider configuredFloorCollider)
        {
            if (configuredBody == null)
            {
                throw new ArgumentNullException(nameof(configuredBody));
            }

            if (configuredCollider == null)
            {
                throw new ArgumentNullException(nameof(configuredCollider));
            }

            if (configuredBody.gameObject != gameObject || configuredCollider.gameObject != gameObject)
            {
                throw new ArgumentException("The ball body and collider must belong to this GameObject.");
            }

            body = configuredBody;
            sphereCollider = configuredCollider;
            floorCollider = configuredFloorCollider;
            ApplyOfficialPhysicsProperties();
            ResetContactObservation();
        }

        public void ResetForKick(
            Transform robotRoot,
            PolicyRole role,
            float floorHeightMeters = 0f)
        {
            if (robotRoot == null)
            {
                throw new ArgumentNullException(nameof(robotRoot));
            }

            float unityRightSign;
            switch (role)
            {
                case PolicyRole.KickLeft:
                    // MuJoCo +Y maps to Tuanjie -X, so the left-kick ball is on
                    // the negative side of the Tuanjie root's right vector.
                    unityRightSign = -1f;
                    break;
                case PolicyRole.KickRight:
                    unityRightSign = 1f;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(role),
                        role,
                        "The skill ball can only be reset for a kick policy.");
            }

            EnsureComponents();
            ApplyOfficialPhysicsProperties();
            gameObject.SetActive(true);
            body.detectCollisions = true;

            Vector3 forward = Vector3.ProjectOnPlane(robotRoot.forward, Vector3.up);
            Vector3 right = Vector3.ProjectOnPlane(robotRoot.right, Vector3.up);
            if (forward.sqrMagnitude < 1e-8f || right.sqrMagnitude < 1e-8f)
            {
                throw new InvalidOperationException("The robot root must define a horizontal forward/right basis.");
            }

            forward.Normalize();
            right.Normalize();
            Vector3 position = robotRoot.position
                + forward * ForwardOffsetMeters
                + right * (unityRightSign * LateralOffsetMeters);
            position.y = floorHeightMeters + RadiusMeters;

            transform.SetPositionAndRotation(position, Quaternion.identity);
            body.position = position;
            body.rotation = Quaternion.identity;
            body.velocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
            ResetContactObservation();
            body.WakeUp();
        }

        public void Hide()
        {
            EnsureComponents();
            body.velocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
            body.detectCollisions = false;
            body.Sleep();
            gameObject.SetActive(false);
        }

        public void SetObservationTime(float timeSeconds)
        {
            if (float.IsNaN(timeSeconds) || float.IsInfinity(timeSeconds))
            {
                throw new ArgumentOutOfRangeException(nameof(timeSeconds), "Observation time must be finite.");
            }

            observationTimeSeconds = timeSeconds;
        }

        public void ResetContactObservation()
        {
            HasNonFloorContact = false;
            FirstNonFloorContactColliderName = string.Empty;
            FirstNonFloorContactTimeSeconds = -1f;
            FirstNonFloorContactPoint = Vector3.zero;
        }

        public void ObserveContact(Collider otherCollider, Vector3 contactPoint, float timeSeconds)
        {
            if (otherCollider == null || otherCollider == floorCollider || HasNonFloorContact)
            {
                return;
            }

            SetObservationTime(timeSeconds);
            HasNonFloorContact = true;
            FirstNonFloorContactColliderName = otherCollider.name;
            FirstNonFloorContactTimeSeconds = timeSeconds;
            FirstNonFloorContactPoint = contactPoint;
        }

        private void Awake()
        {
            EnsureComponents();
            ApplyOfficialPhysicsProperties();
        }

        private void FixedUpdate()
        {
            observationTimeSeconds = Time.fixedTime;
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision == null || collision.collider == null)
            {
                return;
            }

            Vector3 contactPoint = collision.contactCount > 0
                ? collision.GetContact(0).point
                : collision.collider.ClosestPoint(Position);
            ObserveContact(collision.collider, contactPoint, observationTimeSeconds);
        }

        private void EnsureComponents()
        {
            if (body == null)
            {
                body = GetComponent<Rigidbody>();
            }

            if (sphereCollider == null)
            {
                sphereCollider = GetComponent<SphereCollider>();
            }

            if (body == null || sphereCollider == null)
            {
                throw new InvalidOperationException(
                    "MicroDuckSkillBall requires a Rigidbody and SphereCollider.");
            }
        }

        private void ApplyOfficialPhysicsProperties()
        {
            sphereCollider.radius = RadiusMeters;
            sphereCollider.center = Vector3.zero;
            sphereCollider.isTrigger = false;
            body.mass = MassKilograms;
            body.useGravity = true;
            body.isKinematic = false;
            body.drag = 0f;
            body.angularDrag = 0f;
            body.interpolation = RigidbodyInterpolation.None;
            body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            body.inertiaTensorRotation = Quaternion.identity;
            body.inertiaTensor = Vector3.one * MomentOfInertiaKilogramMetersSquared;
            // Unity's default 7 rad/s cap would prevent a 35 mm ball from rolling
            // faster than 0.245 m/s. MuJoCo does not impose that artificial cap.
            body.maxAngularVelocity = 100f;
        }
    }
}

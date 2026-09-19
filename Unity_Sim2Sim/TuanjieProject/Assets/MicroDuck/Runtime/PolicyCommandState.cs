using System;
using UnityEngine;

namespace AgenticRobot.MicroDuck
{
    public sealed class PolicyCommandState
    {
        private readonly float[] twist = new float[3];
        private readonly float[] head = new float[4];
        private readonly float[] body = new float[3];
        private PolicyEntry selected = PolicyCatalog.GetBySlot(2);
        private float phaseStartSeconds;
        private bool phaseTriggered;
        private bool sitRequested;

        public PolicyEntry Selected => selected;

        public void SelectSlot(int slot)
        {
            selected = PolicyCatalog.GetBySlot(slot);
            phaseTriggered = false;
            sitRequested = false;
        }

        public void SetTwist(float forward, float left, float yawRate)
        {
            twist[0] = forward;
            twist[1] = left;
            twist[2] = yawRate;
        }

        public void SetHead(float neckPitch, float headPitch, float headYaw, float headRoll)
        {
            head[0] = neckPitch;
            head[1] = headPitch;
            head[2] = headYaw;
            head[3] = headRoll;
        }

        public void SetBody(float height, float roll, float pitch)
        {
            body[0] = height;
            body[1] = roll;
            body[2] = pitch;
        }

        /// <summary>
        /// Clears transient operator and skill state while keeping the selected
        /// policy. This makes an in-place robot reset equivalent to a fresh
        /// start of the same policy slot.
        /// </summary>
        public void Reset()
        {
            Array.Clear(twist, 0, twist.Length);
            Array.Clear(head, 0, head.Length);
            Array.Clear(body, 0, body.Length);
            phaseStartSeconds = 0f;
            phaseTriggered = false;
            sitRequested = false;
        }

        public void TriggerSkill()
        {
            TriggerSkill(Time.fixedTime);
        }

        public void TriggerSkill(float nowSeconds)
        {
            if (selected.Role == PolicyRole.SitStand)
            {
                sitRequested = !sitRequested;
                return;
            }

            if (selected.UsesPhase)
            {
                phaseStartSeconds = nowSeconds;
                phaseTriggered = true;
            }
        }

        public void Trigger(float nowSeconds)
        {
            TriggerSkill(nowSeconds);
        }

        public void BuildCommand(float nowSeconds, float[] destination)
        {
            if (destination == null)
            {
                throw new ArgumentNullException(nameof(destination));
            }

            if (destination.Length != 13)
            {
                throw new ArgumentException("Policy command must contain 13 values.", nameof(destination));
            }

            Array.Clear(destination, 0, destination.Length);
            if (selected.Role == PolicyRole.SitStand)
            {
                destination[0] = sitRequested ? 1f : 0f;
                return;
            }

            if (selected.UsesPhase)
            {
                float elapsed = phaseTriggered ? Mathf.Max(0f, nowSeconds - phaseStartSeconds) : 0f;
                float phase = Mathf.Min(elapsed / selected.PhasePeriodSeconds, selected.PhaseEnd);
                destination[0] = Mathf.Cos(2f * Mathf.PI * phase);
                destination[1] = Mathf.Sin(2f * Mathf.PI * phase);
                return;
            }

            if (selected.Role == PolicyRole.KickLeft ||
                selected.Role == PolicyRole.KickRight ||
                selected.Role == PolicyRole.Roulade)
            {
                return;
            }

            destination[0] = twist[0];
            destination[1] = twist[1];
            destination[2] = twist[2];
            destination[3] = head[0];
            destination[4] = head[1];
            destination[5] = head[2];
            destination[6] = head[3];
            destination[9] = body[0];
            destination[10] = body[1];
            destination[11] = body[2];
        }
    }
}

using System;

namespace AgenticRobot.MicroDuck
{
    public enum RobotVariant
    {
        Legged,
        Roller,
    }

    public enum PolicyRole
    {
        Walk,
        Stand,
        SitStand,
        GroundPick,
        KickLeft,
        KickRight,
        Roller,
        RollerCrouch,
        Roulade,
    }

    public readonly struct PolicyEntry
    {
        public PolicyEntry(
            int slot,
            string fileName,
            PolicyRole role,
            RobotVariant robotVariant,
            float actionScale,
            float phasePeriodSeconds = 0f,
            float phaseEnd = 0f)
        {
            Slot = slot;
            FileName = fileName;
            Role = role;
            RobotVariant = robotVariant;
            ActionScale = actionScale;
            PhasePeriodSeconds = phasePeriodSeconds;
            PhaseEnd = phaseEnd;
        }

        public int Slot { get; }
        public string FileName { get; }
        public PolicyRole Role { get; }
        public RobotVariant RobotVariant { get; }
        public float ActionScale { get; }
        public float PhasePeriodSeconds { get; }
        public float PhaseEnd { get; }
        public bool UsesPhase => PhasePeriodSeconds > 0f;
    }

    public static class PolicyCatalog
    {
        private static readonly PolicyEntry[] OfficialEntries =
        {
            new PolicyEntry(1, "alpha_walking.onnx", PolicyRole.Walk, RobotVariant.Legged, 1.1f),
            new PolicyEntry(2, "alpha_stand.onnx", PolicyRole.Stand, RobotVariant.Legged, 1f),
            new PolicyEntry(3, "alpha_sitstand.onnx", PolicyRole.SitStand, RobotVariant.Legged, 1f),
            new PolicyEntry(4, "alpha_ground_pick.onnx", PolicyRole.GroundPick, RobotVariant.Legged, 1f, 4f, 0.8f),
            new PolicyEntry(5, "ball_kick_left.onnx", PolicyRole.KickLeft, RobotVariant.Legged, 1f),
            new PolicyEntry(6, "ball_kick_right.onnx", PolicyRole.KickRight, RobotVariant.Legged, 1f),
            new PolicyEntry(7, "roller.onnx", PolicyRole.Roller, RobotVariant.Roller, 0.8f),
            new PolicyEntry(8, "roller_crouch.onnx", PolicyRole.RollerCrouch, RobotVariant.Roller, 0.8f, 5f, 0.7f),
            new PolicyEntry(9, "roulade.onnx", PolicyRole.Roulade, RobotVariant.Legged, 1f),
        };

        public static PolicyEntry[] Entries => (PolicyEntry[])OfficialEntries.Clone();

        public static PolicyEntry GetBySlot(int slot)
        {
            if (slot < 1 || slot > OfficialEntries.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(slot), slot, "Policy slot must be between 1 and 9.");
            }

            return OfficialEntries[slot - 1];
        }

        public static string ToTraceRoleName(PolicyRole role)
        {
            switch (role)
            {
                case PolicyRole.Walk:
                    return "walk";
                case PolicyRole.Stand:
                    return "stand";
                case PolicyRole.SitStand:
                    return "sitstand";
                case PolicyRole.GroundPick:
                    return "ground-pick";
                case PolicyRole.KickLeft:
                    return "kick-left";
                case PolicyRole.KickRight:
                    return "kick-right";
                case PolicyRole.Roller:
                    return "roller";
                case PolicyRole.RollerCrouch:
                    return "roller-crouch";
                case PolicyRole.Roulade:
                    return "roulade";
                default:
                    throw new ArgumentOutOfRangeException(nameof(role), role, "Unknown policy role.");
            }
        }
    }
}

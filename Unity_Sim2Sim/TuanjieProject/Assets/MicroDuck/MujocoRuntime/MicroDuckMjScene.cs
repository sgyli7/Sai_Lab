using Mujoco;
using UnityEngine;

namespace AgenticRobot.MicroDuck.Mujoco
{
    /// <summary>
    /// Registers the official MuJoCo scene singleton before imported MjComponents
    /// receive OnEnable while a serialized scene is loading.
    /// </summary>
    [DefaultExecutionOrder(-2000)]
    public sealed class MicroDuckMjScene : MjScene
    {
    }
}

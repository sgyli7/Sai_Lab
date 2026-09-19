[CmdletBinding()]
param(
    [switch]$ListStages,
    [switch]$DryRun,
    [switch]$Json,
    [switch]$ForceTraining,
    [switch]$CleanGenerated,
    [ValidateSet(
        "all",
        "bootstrap",
        "python-tests",
        "policy-audit",
        "mujoco-rollouts",
        "ppo-smoke",
        "onnx-export",
        "robot-manifest",
        "tuanjie-editmode",
        "tuanjie-playmode",
        "trace-parity",
        "codely-proof",
        "windows-build",
        "environment-acceptance"
    )]
    [string]$Stage = "all",
    [string]$TuanjiePath,
    [string]$TuanjieProject,
    [string]$ArtifactsRoot,
    [string]$CodelyEvidencePath
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Stages = @(
    "bootstrap",
    "python-tests",
    "policy-audit",
    "mujoco-rollouts",
    "ppo-smoke",
    "onnx-export",
    "robot-manifest",
    "tuanjie-editmode",
    "tuanjie-playmode",
    "trace-parity",
    "codely-proof",
    "windows-build",
    "environment-acceptance"
)

$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$DefaultArtifactsRoot = Join-Path $ProjectRoot "artifacts\mvp"
$ArtifactsRootWasExplicit = $PSBoundParameters.ContainsKey("ArtifactsRoot")
if ([string]::IsNullOrWhiteSpace($TuanjieProject)) {
    $TuanjieProject = Join-Path $ProjectRoot "TuanjieProject"
}
else {
    $TuanjieProject = [IO.Path]::GetFullPath($TuanjieProject)
}
if ([string]::IsNullOrWhiteSpace($ArtifactsRoot)) {
    $ArtifactsRoot = $DefaultArtifactsRoot
}
else {
    $ArtifactsRoot = [IO.Path]::GetFullPath($ArtifactsRoot)
}

# A real run must never leave a previous passing verdict behind if preflight
# fails before the stage loop can write a new report. Dry runs remain read-only.
$EarlyRunReportPath = Join-Path $ArtifactsRoot "mvp-report.json"
if (-not $DryRun -and
    -not $ListStages -and
    (Test-Path -LiteralPath $EarlyRunReportPath -PathType Leaf)) {
    Remove-Item -LiteralPath $EarlyRunReportPath -Force
}

if ($CleanGenerated -and $Stage -ne "all") {
    throw "-CleanGenerated is only supported with -Stage all so dependencies are rebuilt in order."
}
if ($CleanGenerated) {
    $workspacePrefix = $ProjectRoot.TrimEnd('\') + '\'
    if (-not $ArtifactsRoot.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a path outside the workspace: $ArtifactsRoot"
    }
}
if ($CleanGenerated -and
    -not $ArtifactsRoot.Equals($DefaultArtifactsRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "-CleanGenerated requires the fixed default artifacts root: $DefaultArtifactsRoot"
}
$StagesWithFixedArtifactContracts = @("all", "tuanjie-editmode", "trace-parity")
if ($ArtifactsRootWasExplicit -and
    $Stage -in $StagesWithFixedArtifactContracts -and
    -not $ArtifactsRoot.Equals($DefaultArtifactsRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Stage '$Stage' requires the fixed default artifacts root because Tuanjie writes signed evidence there."
}
if ([string]::IsNullOrWhiteSpace($CodelyEvidencePath)) {
    $CodelyEvidencePath = Join-Path $ArtifactsRoot "codely\proof.json"
}
else {
    $CodelyEvidencePath = [IO.Path]::GetFullPath($CodelyEvidencePath)
}

function Resolve-Executable {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$Fallbacks = @(),
        [switch]$AllowMissing
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }
    foreach ($fallback in $Fallbacks) {
        if (Test-Path -LiteralPath $fallback -PathType Leaf) {
            return [IO.Path]::GetFullPath($fallback)
        }
    }
    if ($AllowMissing) {
        return $Name
    }
    throw "Required executable '$Name' was not found."
}

function Resolve-TuanjieEditor {
    param([switch]$AllowMissing)

    if (-not [string]::IsNullOrWhiteSpace($TuanjiePath)) {
        $explicit = [IO.Path]::GetFullPath($TuanjiePath)
        if (-not (Test-Path -LiteralPath $explicit -PathType Leaf)) {
            throw "Tuanjie editor does not exist: $explicit"
        }
        return $explicit
    }

    $version = "2022.3.62t14"
    $candidates = @(
        "E:\Program Files\UnityHubEditor\Tuanjie\$version\Editor\Tuanjie.exe",
        "C:\Program Files\UnityHubEditor\Tuanjie\$version\Editor\Tuanjie.exe",
        "D:\Program Files\UnityHubEditor\Tuanjie\$version\Editor\Tuanjie.exe",
        "E:\Program Files\TuanjieHubEditor\$version\Editor\Tuanjie.exe",
        "C:\Program Files\TuanjieHubEditor\$version\Editor\Tuanjie.exe",
        "D:\Program Files\TuanjieHubEditor\$version\Editor\Tuanjie.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    if ($AllowMissing) {
        return "Tuanjie.exe"
    }
    throw "Tuanjie $version was not found. Pass -TuanjiePath explicitly."
}

function New-CommandPlan {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [string]$Condition = "always"
    )

    return [PSCustomObject][ordered]@{
        label = $Label
        executable = $Executable
        arguments = @($Arguments)
        workingDirectory = $WorkingDirectory
        condition = $Condition
    }
}

function New-StagePlan {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][object[]]$Commands,
        [string[]]$Artifacts = @()
    )

    return [PSCustomObject][ordered]@{
        name = $Name
        commands = @($Commands)
        artifacts = @($Artifacts)
    }
}

if ($ListStages) {
    if ($Json) {
        ConvertTo-Json -InputObject $Stages -Compress
    }
    else {
        $Stages
    }
    exit 0
}

$NeedsBootstrapTools = $Stage -in @("all", "bootstrap")
$NeedsTuanjie = $Stage -eq "all" -or $Stage -in @(
    "robot-manifest",
    "tuanjie-editmode",
    "tuanjie-playmode",
    "trace-parity",
    "windows-build"
)
$UvExe = Resolve-Executable -Name "uv" -Fallbacks @(
    "C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
) -AllowMissing:($DryRun -or -not $NeedsBootstrapTools)
$GitExe = Resolve-Executable -Name "git" -AllowMissing:($DryRun -or -not $NeedsBootstrapTools)
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PowerShellExe = Resolve-Executable -Name "powershell.exe" -AllowMissing:$DryRun
$UpstreamRoot = Join-Path $ProjectRoot ".cache\upstream"
$UpstreamLock = Join-Path $ProjectRoot "upstream.lock.json"
$MicroDuckRlRoot = Join-Path $UpstreamRoot "microduck_rl"
$UpstreamPython = Join-Path $MicroDuckRlRoot ".venv\Scripts\python.exe"
$Helper = Join-Path $PSScriptRoot "mvp-gates.py"
$TuanjieExe = Resolve-TuanjieEditor -AllowMissing:($DryRun -or -not $NeedsTuanjie)
$TestRoot = Join-Path $ArtifactsRoot "tuanjie"
$RolloutRoot = Join-Path $ArtifactsRoot "mujoco"
$TrainingReport = Join-Path $ArtifactsRoot "training\ppo-smoke.json"
$TrainingRuntimeReport = Join-Path $ArtifactsRoot "training\ppo-runtime.json"
$TrainingRuntimeValidator = Join-Path $PSScriptRoot "validate-training-runtime.py"
$ExportedOnnx = Join-Path $ProjectRoot ".cache\artifacts\microduck-ppo-64x5.onnx"
$ExportAttestation = Join-Path $ProjectRoot ".cache\artifacts\microduck-ppo-64x5.attestation.json"
$ExportReport = Join-Path $ArtifactsRoot "training\onnx-export.json"
$TuanjieTrace = Join-Path $ProjectRoot "artifacts\mvp\traces\tuanjie-alpha_stand.jsonl"
$TraceReport = Join-Path $ArtifactsRoot "traces\trace-parity.json"
$BarracudaPolicyDirectory = Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Policies\Barracuda"
$WindowsBuild = Join-Path $ProjectRoot "Builds\Windows64\AgenticRobotGame.exe"
$EditModeResults = Join-Path $TestRoot "editmode-results.xml"
$EditModeReport = Join-Path $TestRoot "editmode-report.json"
$NativeBehaviorReport = Join-Path $ArtifactsRoot "tuanjie-native-mujoco-policy-behavior.json"
$NativeBehaviorGate = Join-Path $TestRoot "native-behavior-report.json"
$PlayModeResults = Join-Path $TestRoot "playmode-results.xml"
$PlayModeReport = Join-Path $TestRoot "playmode-report.json"
$PlayModeCameraScreenshot = Join-Path $ProjectRoot "artifacts\visual-acceptance\playmode-camera.png"
$PlayerSmokeReport = Join-Path $TestRoot "player-smoke.json"
$PlayerBundleReport = Join-Path $TestRoot "windows-build-report.json"
$EnvironmentAcceptanceScript = Join-Path $PSScriptRoot "run-visual-acceptance.ps1"
$EnvironmentAcceptanceRoot = Join-Path $ArtifactsRoot "environment-acceptance"
$EnvironmentAcceptanceReport = Join-Path $EnvironmentAcceptanceRoot "latest-report.json"
$EnvironmentAcceptanceValidation = Join-Path $EnvironmentAcceptanceRoot "latest-validation.json"
$RunReportPath = Join-Path $ArtifactsRoot "mvp-report.json"
$WindowsBuildDirectory = Join-Path $ProjectRoot "Builds\Windows64"
$WindowsDataDirectory = Join-Path $WindowsBuildDirectory "AgenticRobotGame_Data"
$CleanArtifactTargets = @(
    (Join-Path $ArtifactsRoot "bootstrap.json"),
    (Join-Path $ArtifactsRoot "python"),
    (Join-Path $ArtifactsRoot "policy-audit.json"),
    (Join-Path $ArtifactsRoot "mujoco"),
    (Join-Path $ArtifactsRoot "training"),
    (Join-Path $ArtifactsRoot "traces"),
    (Join-Path $ArtifactsRoot "tuanjie"),
    (Join-Path $ArtifactsRoot "logs"),
    $EnvironmentAcceptanceRoot,
    (Join-Path $ArtifactsRoot "tuanjie-native-mujoco-policy-behavior.json"),
    (Join-Path $ArtifactsRoot "mvp-report.json")
)
$CleanTargets = @(
    $CleanArtifactTargets
    (Join-Path $TuanjieProject "Library"),
    (Join-Path $TuanjieProject "Temp"),
    (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Prefabs"),
    (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Prefabs.meta"),
    (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\MuJoCo"),
    (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\MuJoCo.meta"),
    (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Scenes"),
    (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Scenes.meta"),
    $WindowsBuildDirectory
)
$CleanPreserved = @((Join-Path $ArtifactsRoot "codely"))
$EditModeRequiredTests = @(
    "AgenticRobot.MicroDuck.Tests.AuthoritativeMujocoPolicyBehaviorTests.AllOfficialPoliciesAndLiveSameModelHotSwapsMeetBehaviorContracts",
    "AgenticRobot.MicroDuck.Tests.BarracudaParityTests.AllOfficialPoliciesMatchOnnxRuntimeReferenceFixtures",
    "AgenticRobot.MicroDuck.Tests.OfficialMujocoPluginTests.NativePluginLoadsAndStepsTheOfficialMicroDuckScene",
    "AgenticRobot.MicroDuck.Tests.OfficialMujocoPrefabImporterTests.ImportsAllExpandedOfficialScenesAsCompleteDeterministicPrefabs",
    "AgenticRobot.MicroDuck.Tests.DemoSceneBuilderTests.CreatesPlayableNativeSceneWithOfficialMuJoCoAndBarracudaControls",
    "AgenticRobot.MicroDuck.Tests.PolicyRuntimeContractTests.OutOfRangeFiniteActionFailsClosedAndClearsEveryAction"
)
$PlayModeRequiredTests = @(
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests.KeyboardPolicySwitchesRunAfterMujocoSceneLateUpdate",
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests.NativeStandPolicyKeepsTheVisibleDuckUprightForFourSeconds",
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests.NativeWalkingPolicyMovesForwardWhileRemainingUprightForSixSeconds",
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests.NativeSceneRunsBarracudaPolicyAndRecreatesOnlyForRobotVariantChanges",
    "AgenticRobot.MicroDuck.Tests.NativeMujocoScenePlayModeTests.NativeRobotRemainsInsideAUsableCameraFrame",
    "AgenticRobot.MicroDuck.Tests.OfficialPolicyPlayModeTests.GeneratedSceneTicksAllNinePoliciesThroughBarracudaAndArticulation",
    "AgenticRobot.MicroDuck.Tests.OfficialPolicyPlayModeTests.HotSwappingRollerPoliciesPreservesLiveRigAndControlState",
    "AgenticRobot.MicroDuck.Tests.KickBallPlayModeTests.BothKickPoliciesStrikeTheBallAndRemainStanding"
)
$PlayModeAllowedSkippedTests = @(
    "AgenticRobot.MicroDuck.Tests.StrictPolicyBehaviorPlayModeTests.OfficialPoliciesMeetMuJoCoDerivedSustainedAndCompoundBehaviorContracts",
    "AgenticRobot.MicroDuck.Tests.OfficialPolicyPlayModeTests.AnkleStepResponseMatchesTheMuJoCoArmatureDynamics",
    "AgenticRobot.MicroDuck.Tests.OfficialPolicyPlayModeTests.DiagnosticHomeDrivesHoldTheRobotWithoutPolicyForTwoSeconds"
)
$EditModeValidationArguments = @(
    $Helper, "tuanjie-results", "--platform", "EditMode",
    "--input", $EditModeResults, "--output", $EditModeReport,
    "--not-before-utc", "<stage-start-utc>"
)
foreach ($requiredTest in $EditModeRequiredTests) {
    $EditModeValidationArguments += @("--required-test", $requiredTest)
}
$PlayModeValidationArguments = @(
    $Helper, "tuanjie-results", "--platform", "PlayMode",
    "--input", $PlayModeResults, "--output", $PlayModeReport,
    "--not-before-utc", "<stage-start-utc>"
)
foreach ($requiredTest in $PlayModeRequiredTests) {
    $PlayModeValidationArguments += @("--required-test", $requiredTest)
}
foreach ($allowedTest in $PlayModeAllowedSkippedTests) {
    $PlayModeValidationArguments += @("--allowed-skipped-test", $allowedTest)
}

function Assert-WorkspaceChildPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $workspace = $ProjectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $resolved = [IO.Path]::GetFullPath($Path).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $prefix = $workspace + [IO.Path]::DirectorySeparatorChar
    if ($resolved.Equals($workspace, [StringComparison]::OrdinalIgnoreCase) -or
        -not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a path outside the workspace: $resolved"
    }
    return $resolved
}

function Assert-NoReparsePointInPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $workspace = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $current = [IO.Path]::GetFullPath($Path)
    while ($current.StartsWith($workspace, [StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing to clean through a reparse point: $current"
            }
        }
        if ($current.Equals($workspace, [StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $parent = Split-Path -Parent $current
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $current) {
            break
        }
        $current = $parent
    }
}

if ($CleanGenerated) {
    foreach ($cleanTarget in $CleanTargets) {
        $null = Assert-WorkspaceChildPath -Path $cleanTarget
        Assert-NoReparsePointInPath -Path $cleanTarget
    }
}

$PlansByName = [ordered]@{}
$UpstreamLockDocument = Get-Content -Raw -LiteralPath (Join-Path $ProjectRoot "upstream.lock.json") |
    ConvertFrom-Json
$BootstrapCommands = @(
    (New-CommandPlan -Label "sync root Python environment" -Executable $UvExe -Arguments @("sync", "--frozen", "--python", "3.12") -WorkingDirectory $ProjectRoot)
)
foreach ($repositoryProperty in $UpstreamLockDocument.repositories.PSObject.Properties) {
    $repositoryName = $repositoryProperty.Name
    $repository = $repositoryProperty.Value
    $repositoryPath = Join-Path $UpstreamRoot $repositoryName
    $BootstrapCommands += New-CommandPlan `
        -Label "verify locked $repositoryName origin" `
        -Executable $GitExe `
        -Arguments @("-C", $repositoryPath, "remote", "get-url", "origin") `
        -WorkingDirectory $ProjectRoot `
        -Condition "clone $($repository.url) when absent; otherwise output must equal $($repository.url)"
    $BootstrapCommands += New-CommandPlan `
        -Label "verify locked $repositoryName commit" `
        -Executable $GitExe `
        -Arguments @("-C", $repositoryPath, "rev-parse", "HEAD") `
        -WorkingDirectory $ProjectRoot `
        -Condition "fetch and detached-checkout $($repository.commit) when needed; output must equal $($repository.commit)"
}
$BootstrapCommands += New-CommandPlan `
    -Label "sync upstream training environment" `
    -Executable $UvExe `
    -Arguments @("sync", "--frozen", "--python", "3.12") `
    -WorkingDirectory $MicroDuckRlRoot `
    -Condition "when upstream .venv is absent"
$BootstrapCommands += New-CommandPlan `
    -Label "verify upstream MuJoCo/Torch runtime" `
    -Executable $UpstreamPython `
    -Arguments @("-c", "import mujoco, torch, warp; assert mujoco.__version__; assert torch.__version__; assert warp.__version__") `
    -WorkingDirectory $MicroDuckRlRoot
$PlansByName["bootstrap"] = New-StagePlan -Name "bootstrap" -Commands $BootstrapCommands `
    -Artifacts @((Join-Path $ArtifactsRoot "bootstrap.json"))
$PlansByName["python-tests"] = New-StagePlan -Name "python-tests" -Commands @(
    (New-CommandPlan -Label "pytest with coverage" -Executable $PythonExe -Arguments @("-m", "pytest", "--cov=agenticrobot_bridge", "--cov-report=json:$ArtifactsRoot\python\coverage.json", "--cov-fail-under=80") -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "ruff check" -Executable $PythonExe -Arguments @("-m", "ruff", "check", "src", "tests", "scripts") -WorkingDirectory $ProjectRoot)
) -Artifacts @((Join-Path $ArtifactsRoot "python\coverage.json"))
$PlansByName["policy-audit"] = New-StagePlan -Name "policy-audit" -Commands @(
    (New-CommandPlan -Label "audit all nine locked ONNX policies" -Executable $PythonExe -Arguments @($Helper, "policy-audit", "--project-root", $ProjectRoot, "--output", (Join-Path $ArtifactsRoot "policy-audit.json")) -WorkingDirectory $ProjectRoot)
) -Artifacts @((Join-Path $ArtifactsRoot "policy-audit.json"))
$PlansByName["mujoco-rollouts"] = New-StagePlan -Name "mujoco-rollouts" -Commands @(
    (New-CommandPlan -Label "run all nine headless MuJoCo scenarios" -Executable $PythonExe -Arguments @($Helper, "mujoco-rollouts", "--project-root", $ProjectRoot, "--output-dir", $RolloutRoot) -WorkingDirectory $ProjectRoot)
) -Artifacts @((Join-Path $RolloutRoot "rollout-report.json"), (Join-Path $RolloutRoot "traces"))
$PlansByName["ppo-smoke"] = New-StagePlan -Name "ppo-smoke" -Commands @(
    (New-CommandPlan -Label "validate exact 64-env x 5-iteration training cache" -Executable $PythonExe -Arguments @($Helper, "training-cache", "--project-root", $ProjectRoot, "--output", $TrainingReport) -WorkingDirectory $ProjectRoot -Condition "reuse when valid unless -ForceTraining"),
    (New-CommandPlan -Label "train exact PPO smoke" -Executable (Join-Path $MicroDuckRlRoot ".venv\Scripts\train.exe") -Arguments @("Mjlab-Velocity-Flat-MicroDuck", "--env.scene.num-envs", "64", "--agent.max-iterations", "5", "--agent.save-interval", "1", "--agent.logger", "tensorboard", "--agent.run-name", "mvp-smoke", "--agent.upload-model", "False", "--enable-nan-guard", "True") -WorkingDirectory $MicroDuckRlRoot -Condition "cache absent/invalid or -ForceTraining"),
    (New-CommandPlan -Label "load PPO checkpoint and TensorBoard evidence" -Executable $UpstreamPython -Arguments @($TrainingRuntimeValidator, "--training-report", $TrainingReport, "--output", $TrainingRuntimeReport) -WorkingDirectory $ProjectRoot)
) -Artifacts @($TrainingReport, $TrainingRuntimeReport)
$PlansByName["onnx-export"] = New-StagePlan -Name "onnx-export" -Commands @(
    (New-CommandPlan -Label "resolve exact PPO checkpoint" -Executable $PythonExe -Arguments @($Helper, "training-cache", "--project-root", $ProjectRoot, "--output", $TrainingReport) -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "official normalized-policy ONNX export" -Executable $UpstreamPython -Arguments @("scripts/export.py", "Mjlab-Velocity-Flat-MicroDuck", "--checkpoint-file", "<checkpoint-from-ppo-smoke.json>", "--onnx-file", $ExportedOnnx, "--device", "cuda:0", "--num-envs", "1") -WorkingDirectory $MicroDuckRlRoot -Condition "validated export absent/invalid or -ForceTraining"),
    (New-CommandPlan -Label "bind ONNX bytes to exact checkpoint bytes" -Executable $PythonExe -Arguments @($Helper, "onnx-attest", "--training-report", $TrainingReport, "--onnx", $ExportedOnnx, "--output", $ExportAttestation) -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "validate exported 61D-to-14D ONNX" -Executable $PythonExe -Arguments @($Helper, "onnx-export", "--project-root", $ProjectRoot, "--training-report", $TrainingReport, "--onnx", $ExportedOnnx, "--attestation", $ExportAttestation, "--output", $ExportReport) -WorkingDirectory $ProjectRoot)
) -Artifacts @($ExportedOnnx, $ExportAttestation, $ExportReport)
$PlansByName["robot-manifest"] = New-StagePlan -Name "robot-manifest" -Commands @(
    (New-CommandPlan -Label "generate deterministic Tuanjie assets" -Executable $PythonExe -Arguments @("-m", "agenticrobot_bridge.tuanjie_assets", "--project-root", $ProjectRoot, "--tuanjie-project", $TuanjieProject) -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "import both robots and rebuild the PhysX and native MuJoCo playable scenes" -Executable $TuanjieExe -Arguments @("-batchmode", "-nographics", "-quit", "-projectPath", $TuanjieProject, "-executeMethod", "AgenticRobot.MicroDuck.Editor.DemoSceneBuilder.CreateAllSceneAssets", "-logFile", (Join-Path $TestRoot "prefab-import.log")) -WorkingDirectory $ProjectRoot)
) -Artifacts @((Join-Path $TuanjieProject "Assets\MicroDuck\Generated\asset-report.json"), (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Prefabs\MicroDuck-Legged.prefab"), (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Prefabs\MicroDuck-Roller.prefab"), (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\MuJoCo\Legged\MicroDuck-MuJoCo-Legged.prefab"), (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\MuJoCo\Roller\MicroDuck-MuJoCo-Roller.prefab"), (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\MuJoCo\Ball\MicroDuck-MuJoCo-Ball.prefab"), (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Scenes\MicroDuckMvp.unity"), (Join-Path $TuanjieProject "Assets\MicroDuck\Generated\Scenes\MicroDuckNativeMvp.unity"))
$PlansByName["tuanjie-editmode"] = New-StagePlan -Name "tuanjie-editmode" -Commands @(
    (New-CommandPlan -Label "run Tuanjie EditMode tests" -Executable $TuanjieExe -Arguments @("-batchmode", "-nographics", "-projectPath", $TuanjieProject, "-runTests", "-testPlatform", "EditMode", "-testResults", $EditModeResults, "-logFile", (Join-Path $TestRoot "editmode.log")) -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "validate identity and freshness of Tuanjie EditMode results" -Executable $PythonExe -Arguments $EditModeValidationArguments -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "validate all native MuJoCo behavior evidence" -Executable $PythonExe -Arguments @($Helper, "native-behavior", "--input", $NativeBehaviorReport, "--output", $NativeBehaviorGate, "--not-before-utc", "<stage-start-utc>") -WorkingDirectory $ProjectRoot)
) -Artifacts @($EditModeResults, $EditModeReport, $NativeBehaviorReport, $NativeBehaviorGate, (Join-Path $TestRoot "editmode.log"))
$PlansByName["tuanjie-playmode"] = New-StagePlan -Name "tuanjie-playmode" -Commands @(
    (New-CommandPlan -Label "run Tuanjie PlayMode tests with the real graphics device" -Executable $TuanjieExe -Arguments @("-batchmode", "-projectPath", $TuanjieProject, "-runTests", "-testPlatform", "PlayMode", "-testResults", $PlayModeResults, "-logFile", (Join-Path $TestRoot "playmode.log")) -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "validate identity and freshness of Tuanjie PlayMode results" -Executable $PythonExe -Arguments $PlayModeValidationArguments -WorkingDirectory $ProjectRoot)
) -Artifacts @($PlayModeResults, $PlayModeReport, $PlayModeCameraScreenshot, (Join-Path $TestRoot "playmode.log"))
$PlansByName["trace-parity"] = New-StagePlan -Name "trace-parity" -Commands @(
    (New-CommandPlan -Label "export one real Tuanjie Barracuda policy tick" -Executable $TuanjieExe -Arguments @("-batchmode", "-nographics", "-quit", "-projectPath", $TuanjieProject, "-executeMethod", "AgenticRobot.MicroDuck.Editor.TraceBatchExporter.ExportOnePolicyBatch", "-logFile", (Join-Path $TestRoot "trace-export.log")) -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "audit Tuanjie trace against its Barracuda ONNX" -Executable $PythonExe -Arguments @($Helper, "trace-parity", "--trace", $TuanjieTrace, "--policy-directory", $BarracudaPolicyDirectory, "--output", $TraceReport) -WorkingDirectory $ProjectRoot)
) -Artifacts @($TuanjieTrace, $TraceReport, (Join-Path $TestRoot "trace-export.log"))
$PlansByName["codely-proof"] = New-StagePlan -Name "codely-proof" -Commands @(
    (New-CommandPlan -Label "verify independently-created Codely evidence" -Executable $PythonExe -Arguments @($Helper, "evidence", "--kind", "codely", "--input", $CodelyEvidencePath, "--output", (Join-Path $ArtifactsRoot "codely\gate.json")) -WorkingDirectory $ProjectRoot)
) -Artifacts @($CodelyEvidencePath, (Join-Path $ArtifactsRoot "codely\gate.json"))
$PlansByName["windows-build"] = New-StagePlan -Name "windows-build" -Commands @(
    (New-CommandPlan -Label "build Windows x64 player" -Executable $TuanjieExe -Arguments @("-batchmode", "-nographics", "-quit", "-projectPath", $TuanjieProject, "-executeMethod", "AgenticRobot.MicroDuck.Editor.DemoSceneBuilder.BuildWindows64", "-logFile", (Join-Path $TestRoot "windows-build.log")) -WorkingDirectory $ProjectRoot),
    (New-CommandPlan -Label "smoke native MuJoCo Windows player" -Executable $WindowsBuild -Arguments @("-batchmode", "-nographics", "-logFile", (Join-Path $TestRoot "windows-player.log"), "-microduckSmokeReport", $PlayerSmokeReport) -WorkingDirectory $WindowsBuildDirectory),
    (New-CommandPlan -Label "validate complete Windows player bundle, native provenance, and smoke" -Executable $PythonExe -Arguments @($Helper, "player-smoke", "--build-directory", $WindowsBuildDirectory, "--input", $PlayerSmokeReport, "--output", $PlayerBundleReport, "--upstream-lock", $UpstreamLock) -WorkingDirectory $ProjectRoot)
) -Artifacts @($WindowsBuild, (Join-Path $WindowsBuildDirectory "TuanjiePlayer.dll"), (Join-Path $WindowsDataDirectory "Plugins\x86_64\mujoco.dll"), $PlayerSmokeReport, $PlayerBundleReport, (Join-Path $TestRoot "windows-build.log"), (Join-Path $TestRoot "windows-player.log"))
$PlansByName["environment-acceptance"] = New-StagePlan -Name "environment-acceptance" -Commands @(
    (New-CommandPlan -Label "capture and validate the complete interactive environment tour" -Executable $PowerShellExe -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $EnvironmentAcceptanceScript, "-PlayerPath", $WindowsBuild, "-OutputDirectory", $EnvironmentAcceptanceRoot) -WorkingDirectory $ProjectRoot)
) -Artifacts @($EnvironmentAcceptanceReport, $EnvironmentAcceptanceValidation)

$SelectedNames = if ($Stage -eq "all") { $Stages } else { @($Stage) }
$SelectedPlans = @($SelectedNames | ForEach-Object { $PlansByName[$_] })
$PlanDocument = [PSCustomObject][ordered]@{
    schemaVersion = 1
    selectedStage = $Stage
    dryRun = [bool]$DryRun
    forceTraining = [bool]$ForceTraining
    projectRoot = $ProjectRoot
    tuanjieProject = $TuanjieProject
    tuanjieEditor = $TuanjieExe
    artifactsRoot = $ArtifactsRoot
    clean = [PSCustomObject][ordered]@{
        requested = [bool]$CleanGenerated
        performed = $false
        targets = @($CleanTargets)
        preserved = @($CleanPreserved)
    }
    stages = $SelectedPlans
}

if ($DryRun) {
    if ($Json) {
        $PlanDocument | ConvertTo-Json -Depth 8 -Compress
    }
    else {
        foreach ($stagePlan in $SelectedPlans) {
            Write-Output "[$($stagePlan.name)]"
            foreach ($commandPlan in $stagePlan.commands) {
                Write-Output "  $($commandPlan.label): $($commandPlan.executable) $($commandPlan.arguments -join ' ')"
            }
        }
    }
    exit 0
}

function Remove-SafeGeneratedPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = Assert-WorkspaceChildPath -Path $Path
    Assert-NoReparsePointInPath -Path $resolved
    if (Test-Path -LiteralPath $resolved) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

function Remove-ExactOutputFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (Test-Path -LiteralPath $Path -PathType Container) {
        throw "Refusing to replace an output file because it is a directory: $Path"
    }
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $item = Get-Item -LiteralPath $Path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing to replace a reparse-point output file: $Path"
        }
        Remove-Item -LiteralPath $Path -Force
    }
}

function Invoke-CleanGenerated {
    foreach ($target in $CleanTargets) {
        Remove-SafeGeneratedPath -Path $target
    }
}

$CleanPerformed = $false
if ($CleanGenerated) {
    Invoke-CleanGenerated
    $CleanPerformed = $true
}

function Invoke-CommandPlan {
    param(
        [Parameter(Mandatory = $true)][object]$Command,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    if (-not (Test-Path -LiteralPath $Command.executable -PathType Leaf)) {
        $resolved = Get-Command $Command.executable -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $resolved) {
            throw "Executable for '$($Command.label)' was not found: $($Command.executable)"
        }
    }
    if (-not (Test-Path -LiteralPath $Command.workingDirectory -PathType Container)) {
        throw "Working directory for '$($Command.label)' does not exist: $($Command.workingDirectory)"
    }

    $logDirectory = Split-Path -Parent $LogPath
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    Push-Location $Command.workingDirectory
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        if ([IO.Path]::GetFileName($Command.executable).Equals(
            "Tuanjie.exe",
            [StringComparison]::OrdinalIgnoreCase
        ) -or [IO.Path]::GetFileName($Command.executable).Equals(
            "AgenticRobotGame.exe",
            [StringComparison]::OrdinalIgnoreCase
        )) {
            # Tuanjie detaches from a PowerShell invocation on Windows. Waiting on
            # the returned process is required before a following gate reads its
            # XML/log/artifact outputs or starts another editor against the project.
            $process = Start-Process `
                -FilePath $Command.executable `
                -ArgumentList @($Command.arguments) `
                -WorkingDirectory $Command.workingDirectory `
                -WindowStyle Hidden `
                -PassThru
            $isPlayer = [IO.Path]::GetFileName($Command.executable).Equals(
                "AgenticRobotGame.exe",
                [StringComparison]::OrdinalIgnoreCase
            )
            $completedInTime = if ($isPlayer) {
                $process.WaitForExit(120000)
            } else {
                $process.WaitForExit(1800000)
            }
            if (-not $completedInTime) {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                if (-not $isPlayer) {
                    throw "Tuanjie command '$($Command.label)' timed out after 30 minutes."
                }
                throw "Process '$($Command.label)' timed out after 120 seconds."
            }
            $process.Refresh()
            $exitCode = $process.ExitCode
            $captured = @("Tuanjie process exited with code $exitCode.")
        }
        else {
            $captured = @(& $Command.executable @($Command.arguments) 2>&1)
            $exitCode = $LASTEXITCODE
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Pop-Location
    }
    $captured | Out-File -LiteralPath $LogPath -Encoding utf8
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    if ($exitCode -ne 0) {
        $tail = @($captured | Select-Object -Last 20) -join [Environment]::NewLine
        throw "Command '$($Command.label)' failed with exit code $exitCode.$([Environment]::NewLine)$tail"
    }
}

function Invoke-NativeText {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $captured = @(& $Executable @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Pop-Location
    }
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $Executable $($Arguments -join ' ')"
    }
    return (($captured | Out-String).Trim())
}

function Assert-PassingJsonArtifact {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required JSON artifact was not produced: $Path"
    }
    $document = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    if ($document.passed -ne $true) {
        throw "JSON artifact did not pass its acceptance gate: $Path"
    }
    return $document
}

function Write-JsonDocument {
    param(
        [Parameter(Mandatory = $true)][object]$Document,
        [Parameter(Mandatory = $true)][string]$Path,
        [int]$Depth = 8
    )

    $directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $contents = ($Document | ConvertTo-Json -Depth $Depth) + [Environment]::NewLine
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($Path, $contents, $encoding)
}

New-Item -ItemType Directory -Path $ArtifactsRoot -Force | Out-Null
New-Item -ItemType Directory -Path $TestRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $ArtifactsRoot "python") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $ArtifactsRoot "training") -Force | Out-Null
if (Test-Path -LiteralPath $RunReportPath -PathType Leaf) {
    Remove-Item -LiteralPath $RunReportPath -Force
}
$RunStarted = [DateTimeOffset]::UtcNow
$StageResults = @()
$RunError = $null
foreach ($stagePlan in $SelectedPlans) {
    $stageStarted = [DateTimeOffset]::UtcNow
    $stageStatus = "passed"
    try {
        switch ($stagePlan.name) {
            "bootstrap" {
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\bootstrap-root-uv.log"
                )
                $nativeBinaryLock = $UpstreamLockDocument.nativeBinaries.mujocoWindowsX64
                if ($null -eq $nativeBinaryLock) {
                    throw "upstream.lock.json has no mujocoWindowsX64 native binary contract."
                }
                $expectedNativeHash = ([string]$nativeBinaryLock.sha256).ToLowerInvariant()
                if ($expectedNativeHash -notmatch '^[0-9a-f]{64}$') {
                    throw "Locked MuJoCo native SHA-256 is invalid."
                }
                $rootUvLock = Join-Path $ProjectRoot "uv.lock"
                $rootUvLockText = Get-Content -Raw -LiteralPath $rootUvLock
                if (-not $rootUvLockText.Contains([string]$nativeBinaryLock.url) -or
                    -not $rootUvLockText.Contains([string]$nativeBinaryLock.archiveSha256)) {
                    throw "uv.lock does not bind the official MuJoCo wheel URL and archive hash."
                }
                $installedNativePath = Join-Path $ProjectRoot ".venv\Lib\site-packages\mujoco\mujoco.dll"
                $projectNativePath = [IO.Path]::GetFullPath((
                    Join-Path $ProjectRoot ([string]$nativeBinaryLock.projectPath)
                ))
                $null = Assert-WorkspaceChildPath -Path $projectNativePath
                foreach ($nativePath in @($installedNativePath, $projectNativePath)) {
                    if (-not (Test-Path -LiteralPath $nativePath -PathType Leaf)) {
                        throw "Locked MuJoCo native DLL is missing: $nativePath"
                    }
                    $actualNativeHash = (Get-FileHash -LiteralPath $nativePath -Algorithm SHA256).Hash.ToLowerInvariant()
                    if ($actualNativeHash -ne $expectedNativeHash) {
                        throw "MuJoCo native DLL hash mismatch at ${nativePath}: $actualNativeHash"
                    }
                }
                $nativeBinaryEvidence = [PSCustomObject][ordered]@{
                    version = [string]$nativeBinaryLock.version
                    distribution = [string]$nativeBinaryLock.distribution
                    url = [string]$nativeBinaryLock.url
                    archiveSha256 = [string]$nativeBinaryLock.archiveSha256
                    archiveMember = [string]$nativeBinaryLock.archiveMember
                    sha256 = $expectedNativeHash
                    installedPath = [IO.Path]::GetFullPath($installedNativePath)
                    projectPath = $projectNativePath
                }
                New-Item -ItemType Directory -Path $UpstreamRoot -Force | Out-Null
                $repositoryEvidence = @()
                foreach ($repositoryProperty in $UpstreamLockDocument.repositories.PSObject.Properties) {
                    $repositoryName = $repositoryProperty.Name
                    $repository = $repositoryProperty.Value
                    $repositoryPath = Join-Path $UpstreamRoot $repositoryName
                    if (-not (Test-Path -LiteralPath (Join-Path $repositoryPath ".git") -PathType Container)) {
                        if (Test-Path -LiteralPath $repositoryPath) {
                            $existing = @(Get-ChildItem -LiteralPath $repositoryPath -Force)
                            if ($existing.Count -gt 0) {
                                throw "Refusing to clone into non-empty non-Git path: $repositoryPath"
                            }
                        }
                        $clone = New-CommandPlan -Label "clone $repositoryName" `
                            -Executable $GitExe `
                            -Arguments @("clone", "--no-checkout", [string]$repository.url, $repositoryPath) `
                            -WorkingDirectory $ProjectRoot
                        Invoke-CommandPlan -Command $clone -LogPath (
                            Join-Path $ArtifactsRoot "logs\bootstrap-$repositoryName-clone.log"
                        )
                    }

                    $actualOrigin = Invoke-NativeText -Executable $GitExe `
                        -Arguments @("-C", $repositoryPath, "remote", "get-url", "origin") `
                        -WorkingDirectory $ProjectRoot
                    if ($actualOrigin -ne [string]$repository.url) {
                        throw "Locked repository $repositoryName has origin '$actualOrigin', expected '$($repository.url)'."
                    }
                    $actualCommit = Invoke-NativeText -Executable $GitExe `
                        -Arguments @("-C", $repositoryPath, "rev-parse", "HEAD") `
                        -WorkingDirectory $ProjectRoot
                    $trackedChanges = Invoke-NativeText -Executable $GitExe `
                        -Arguments @("-C", $repositoryPath, "status", "--porcelain") `
                        -WorkingDirectory $ProjectRoot
                    if (-not [string]::IsNullOrWhiteSpace($trackedChanges)) {
                        throw "Locked repository $repositoryName has local changes; refusing to use a modified upstream tree."
                    }
                    if ($actualCommit -ne [string]$repository.commit) {
                        $fetch = New-CommandPlan -Label "fetch locked $repositoryName commit" `
                            -Executable $GitExe `
                            -Arguments @("-C", $repositoryPath, "fetch", "origin", [string]$repository.commit) `
                            -WorkingDirectory $ProjectRoot
                        Invoke-CommandPlan -Command $fetch -LogPath (
                            Join-Path $ArtifactsRoot "logs\bootstrap-$repositoryName-fetch.log"
                        )
                        $checkout = New-CommandPlan -Label "checkout locked $repositoryName commit" `
                            -Executable $GitExe `
                            -Arguments @("-C", $repositoryPath, "checkout", "--detach", [string]$repository.commit) `
                            -WorkingDirectory $ProjectRoot
                        Invoke-CommandPlan -Command $checkout -LogPath (
                            Join-Path $ArtifactsRoot "logs\bootstrap-$repositoryName-checkout.log"
                        )
                        $actualCommit = Invoke-NativeText -Executable $GitExe `
                            -Arguments @("-C", $repositoryPath, "rev-parse", "HEAD") `
                            -WorkingDirectory $ProjectRoot
                    }
                    if ($actualCommit -ne [string]$repository.commit) {
                        throw "Repository $repositoryName is at $actualCommit, expected $($repository.commit)."
                    }
                    $repositoryEvidence += [PSCustomObject][ordered]@{
                        name = $repositoryName
                        url = $actualOrigin
                        commit = $actualCommit
                        path = [IO.Path]::GetFullPath($repositoryPath)
                    }
                }

                if (-not (Test-Path -LiteralPath $UpstreamPython -PathType Leaf)) {
                    $upstreamSync = New-CommandPlan -Label "sync upstream training environment" `
                        -Executable $UvExe `
                        -Arguments @("sync", "--frozen", "--python", "3.12") `
                        -WorkingDirectory $MicroDuckRlRoot
                    Invoke-CommandPlan -Command $upstreamSync -LogPath (
                        Join-Path $ArtifactsRoot "logs\bootstrap-upstream-uv.log"
                    )
                }

                $hasNvidia = $null -ne (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)
                $cudaReady = $false
                if ($hasNvidia -and (Test-Path -LiteralPath $UpstreamPython -PathType Leaf)) {
                    try {
                        $null = Invoke-NativeText -Executable $UpstreamPython `
                            -Arguments @("-c", "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)") `
                            -WorkingDirectory $MicroDuckRlRoot
                        $cudaReady = $true
                    }
                    catch {
                        $cudaReady = $false
                    }
                }
                if ($hasNvidia -and -not $cudaReady) {
                    $torchWheel = Join-Path $ProjectRoot ".cache\downloads\torch-2.9.1+cu128-cp312-cp312-win_amd64.whl"
                    $expectedWheelHash = "3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0"
                    if ((Test-Path -LiteralPath $torchWheel -PathType Leaf) -and
                        (Get-FileHash -LiteralPath $torchWheel -Algorithm SHA256).Hash.ToLowerInvariant() -eq $expectedWheelHash) {
                        $cudaInstallArguments = @(
                            "pip", "install", "--python", $UpstreamPython,
                            "--reinstall", "--no-deps", $torchWheel
                        )
                        $cudaInstall = New-CommandPlan -Label "install cached CUDA Torch wheel" `
                            -Executable $UvExe -Arguments $cudaInstallArguments `
                            -WorkingDirectory $MicroDuckRlRoot
                        Invoke-CommandPlan -Command $cudaInstall -LogPath (
                            Join-Path $ArtifactsRoot "logs\bootstrap-cuda-torch.log"
                        )
                        $visionInstall = New-CommandPlan -Label "install CUDA torchvision" `
                            -Executable $UvExe `
                            -Arguments @("pip", "install", "--python", $UpstreamPython, "--reinstall", "--no-deps", "torchvision==0.24.1+cu128", "--index-url", "https://download.pytorch.org/whl/cu128") `
                            -WorkingDirectory $MicroDuckRlRoot
                        Invoke-CommandPlan -Command $visionInstall -LogPath (
                            Join-Path $ArtifactsRoot "logs\bootstrap-cuda-vision.log"
                        )
                    }
                    else {
                        $cudaInstall = New-CommandPlan -Label "install official CUDA Torch runtime" `
                            -Executable $UvExe `
                            -Arguments @("pip", "install", "--python", $UpstreamPython, "--reinstall", "torch==2.9.1+cu128", "torchvision==0.24.1+cu128", "--index-url", "https://download.pytorch.org/whl/cu128") `
                            -WorkingDirectory $MicroDuckRlRoot
                        Invoke-CommandPlan -Command $cudaInstall -LogPath (
                            Join-Path $ArtifactsRoot "logs\bootstrap-cuda-download.log"
                        )
                    }
                }

                Invoke-CommandPlan -Command $stagePlan.commands[-1] -LogPath (
                    Join-Path $ArtifactsRoot "logs\bootstrap-runtime-probe.log"
                )

                $runtimeJson = Invoke-NativeText -Executable $UpstreamPython `
                    -Arguments @("-c", "import json,mujoco,torch;print(json.dumps({'mujoco':mujoco.__version__,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))") `
                    -WorkingDirectory $MicroDuckRlRoot
                $runtime = $runtimeJson | ConvertFrom-Json
                if ($hasNvidia -and $runtime.cuda -ne $true) {
                    throw "NVIDIA GPU is present but the upstream Torch runtime cannot use CUDA."
                }
                [PSCustomObject][ordered]@{
                    schemaVersion = 1
                    passed = $true
                    repositories = $repositoryEvidence
                    rootPython = $PythonExe
                    upstreamPython = $UpstreamPython
                    runtime = $runtime
                    nativeBinary = $nativeBinaryEvidence
                } | ForEach-Object {
                    Write-JsonDocument -Document $_ `
                        -Path (Join-Path $ArtifactsRoot "bootstrap.json")
                }
            }
            "python-tests" {
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\pytest.log"
                )
                Invoke-CommandPlan -Command $stagePlan.commands[1] -LogPath (
                    Join-Path $ArtifactsRoot "logs\ruff.log"
                )
            }
            "policy-audit" {
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\policy-audit.log"
                )
            }
            "mujoco-rollouts" {
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\mujoco-rollouts.log"
                )
            }
            "ppo-smoke" {
                $needsTraining = [bool]$ForceTraining
                if (-not $needsTraining) {
                    try {
                        Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                            Join-Path $ArtifactsRoot "logs\ppo-cache.log"
                        )
                        $stageStatus = "cached"
                    }
                    catch {
                        $needsTraining = $true
                    }
                }
                if ($needsTraining) {
                    $trainingCommandStartedUtc = [DateTime]::UtcNow
                    Invoke-CommandPlan -Command $stagePlan.commands[1] -LogPath (
                        Join-Path $ArtifactsRoot "logs\ppo-training.log"
                    )
                    Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                        Join-Path $ArtifactsRoot "logs\ppo-cache-after-training.log"
                    )
                    $freshTraining = Get-Content -Raw -LiteralPath $TrainingReport |
                        ConvertFrom-Json
                    $freshCheckpoint = Get-Item -LiteralPath ([string]$freshTraining.checkpoint)
                    if ($freshCheckpoint.LastWriteTimeUtc -lt $trainingCommandStartedUtc.AddSeconds(-2)) {
                        throw "PPO command returned success but did not produce a fresh model_4.pt checkpoint."
                    }
                }
                Invoke-CommandPlan -Command $stagePlan.commands[2] -LogPath (
                    Join-Path $ArtifactsRoot "logs\ppo-runtime-validation.log"
                )
            }
            "onnx-export" {
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\onnx-training-cache.log"
                )
                $training = Get-Content -Raw -LiteralPath $TrainingReport | ConvertFrom-Json
                $exportArguments = @($stagePlan.commands[1].arguments)
                $checkpointIndex = [Array]::IndexOf($exportArguments, "<checkpoint-from-ppo-smoke.json>")
                if ($checkpointIndex -lt 0) {
                    throw "The ONNX export plan has no checkpoint placeholder."
                }
                $exportArguments[$checkpointIndex] = [string]$training.checkpoint
                $exportCommand = New-CommandPlan -Label $stagePlan.commands[1].label `
                    -Executable $stagePlan.commands[1].executable `
                    -Arguments $exportArguments `
                    -WorkingDirectory $stagePlan.commands[1].workingDirectory `
                    -Condition $stagePlan.commands[1].condition

                $needsExport = [bool]$ForceTraining
                if (-not $needsExport) {
                    try {
                        Invoke-CommandPlan -Command $stagePlan.commands[3] -LogPath (
                            Join-Path $ArtifactsRoot "logs\onnx-cache.log"
                        )
                        $stageStatus = "cached"
                    }
                    catch {
                        $needsExport = $true
                    }
                }
                if ($needsExport) {
                    Invoke-CommandPlan -Command $exportCommand -LogPath (
                        Join-Path $ArtifactsRoot "logs\onnx-export.log"
                    )
                    Invoke-CommandPlan -Command $stagePlan.commands[2] -LogPath (
                        Join-Path $ArtifactsRoot "logs\onnx-attestation.log"
                    )
                    Invoke-CommandPlan -Command $stagePlan.commands[3] -LogPath (
                        Join-Path $ArtifactsRoot "logs\onnx-validation.log"
                    )
                }
            }
            "codely-proof" {
                Remove-ExactOutputFile -Path (Join-Path $ArtifactsRoot "codely\gate.json")
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\codely-proof.log"
                )
            }
            "trace-parity" {
                Remove-SafeGeneratedPath -Path $TuanjieTrace
                Remove-SafeGeneratedPath -Path $TraceReport
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\trace-export-process.log"
                )
                Invoke-CommandPlan -Command $stagePlan.commands[1] -LogPath (
                    Join-Path $ArtifactsRoot "logs\trace-parity.log"
                )
            }
            "robot-manifest" {
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\robot-assets.log"
                )
                $null = Assert-PassingJsonArtifact -Path $stagePlan.artifacts[0]
                Invoke-CommandPlan -Command $stagePlan.commands[1] -LogPath (
                    Join-Path $ArtifactsRoot "logs\prefab-import-process.log"
                )
                foreach ($requiredAsset in @($stagePlan.artifacts | Select-Object -Skip 1)) {
                    if (-not (Test-Path -LiteralPath $requiredAsset -PathType Leaf)) {
                        throw "Robot and scene import did not produce: $requiredAsset"
                    }
                }
            }
            { $_ -in @("tuanjie-editmode", "tuanjie-playmode") } {
                $resultPath = if ($stagePlan.name -eq "tuanjie-editmode") {
                    $EditModeResults
                } else {
                    $PlayModeResults
                }
                $reportPath = if ($stagePlan.name -eq "tuanjie-editmode") {
                    $EditModeReport
                } else {
                    $PlayModeReport
                }
                Remove-SafeGeneratedPath -Path $resultPath
                Remove-SafeGeneratedPath -Path $reportPath
                if ($stagePlan.name -eq "tuanjie-editmode") {
                    Remove-SafeGeneratedPath -Path $NativeBehaviorReport
                    Remove-SafeGeneratedPath -Path $NativeBehaviorGate
                } else {
                    Remove-SafeGeneratedPath -Path $PlayModeCameraScreenshot
                }
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\$($stagePlan.name)-process.log"
                )
                $validationArguments = @($stagePlan.commands[1].arguments)
                $timestampIndex = [Array]::IndexOf($validationArguments, "<stage-start-utc>")
                if ($timestampIndex -ge 0) {
                    $validationArguments[$timestampIndex] = $stageStarted.UtcDateTime.ToString("o")
                }
                $validationCommand = New-CommandPlan -Label $stagePlan.commands[1].label `
                    -Executable $stagePlan.commands[1].executable `
                    -Arguments $validationArguments `
                    -WorkingDirectory $stagePlan.commands[1].workingDirectory
                Invoke-CommandPlan -Command $validationCommand -LogPath (
                    Join-Path $ArtifactsRoot "logs\$($stagePlan.name)-validation.log"
                )
                if ($stagePlan.name -eq "tuanjie-editmode") {
                    $behaviorArguments = @($stagePlan.commands[2].arguments)
                    $behaviorTimestampIndex = [Array]::IndexOf(
                        $behaviorArguments,
                        "<stage-start-utc>"
                    )
                    if ($behaviorTimestampIndex -ge 0) {
                        $behaviorArguments[$behaviorTimestampIndex] = `
                            $stageStarted.UtcDateTime.ToString("o")
                    }
                    $behaviorCommand = New-CommandPlan -Label $stagePlan.commands[2].label `
                        -Executable $stagePlan.commands[2].executable `
                        -Arguments $behaviorArguments `
                        -WorkingDirectory $stagePlan.commands[2].workingDirectory
                    Invoke-CommandPlan -Command $behaviorCommand -LogPath (
                        Join-Path $ArtifactsRoot "logs\native-behavior-validation.log"
                    )
                }
            }
            "windows-build" {
                Remove-SafeGeneratedPath -Path $WindowsBuildDirectory
                Remove-SafeGeneratedPath -Path $PlayerSmokeReport
                Remove-SafeGeneratedPath -Path $PlayerBundleReport
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\windows-build-process.log"
                )
                if (-not (Test-Path -LiteralPath $WindowsBuild -PathType Leaf)) {
                    throw "Windows x64 build did not produce: $WindowsBuild"
                }
                Invoke-CommandPlan -Command $stagePlan.commands[1] -LogPath (
                    Join-Path $ArtifactsRoot "logs\windows-player-process.log"
                )
                Invoke-CommandPlan -Command $stagePlan.commands[2] -LogPath (
                    Join-Path $ArtifactsRoot "logs\windows-player-validation.log"
                )
            }
            "environment-acceptance" {
                Invoke-CommandPlan -Command $stagePlan.commands[0] -LogPath (
                    Join-Path $ArtifactsRoot "logs\environment-acceptance.log"
                )
                foreach ($requiredEvidence in $stagePlan.artifacts) {
                    if (-not (Test-Path -LiteralPath $requiredEvidence -PathType Leaf)) {
                        throw "Environment acceptance did not produce: $requiredEvidence"
                    }
                }
                $environmentValidation = Get-Content -Raw -LiteralPath $EnvironmentAcceptanceValidation |
                    ConvertFrom-Json
                if ($environmentValidation.valid -ne $true) {
                    throw "Environment acceptance evidence did not validate: $EnvironmentAcceptanceValidation"
                }
            }
            default {
                throw "Execution for stage '$($stagePlan.name)' has not been implemented."
            }
        }
        $stageFinished = [DateTimeOffset]::UtcNow
        $StageResults += [PSCustomObject][ordered]@{
            name = $stagePlan.name
            status = $stageStatus
            startedUtc = $stageStarted.ToString("o")
            finishedUtc = $stageFinished.ToString("o")
            durationSeconds = [Math]::Round(($stageFinished - $stageStarted).TotalSeconds, 3)
            artifacts = @($stagePlan.artifacts)
        }
    }
    catch {
        $stageFinished = [DateTimeOffset]::UtcNow
        $StageResults += [PSCustomObject][ordered]@{
            name = $stagePlan.name
            status = "failed"
            startedUtc = $stageStarted.ToString("o")
            finishedUtc = $stageFinished.ToString("o")
            durationSeconds = [Math]::Round(($stageFinished - $stageStarted).TotalSeconds, 3)
            artifacts = @($stagePlan.artifacts)
            error = $_.Exception.Message
        }
        $RunError = $_.Exception.Message
        break
    }
}

$RunFinished = [DateTimeOffset]::UtcNow
$RunReport = [PSCustomObject][ordered]@{
    schemaVersion = 1
    passed = $null -eq $RunError
    selectedStage = $Stage
    forceTraining = [bool]$ForceTraining
    startedUtc = $RunStarted.ToString("o")
    finishedUtc = $RunFinished.ToString("o")
    durationSeconds = [Math]::Round(($RunFinished - $RunStarted).TotalSeconds, 3)
    clean = [PSCustomObject][ordered]@{
        requested = [bool]$CleanGenerated
        performed = $CleanPerformed
        targets = @($CleanTargets)
        preserved = @($CleanPreserved)
    }
    stages = $StageResults
    error = $RunError
}
Write-JsonDocument -Document $RunReport -Path $RunReportPath
if ($Json) {
    $RunReport | ConvertTo-Json -Depth 8 -Compress
}
else {
    foreach ($stageResult in $StageResults) {
        Write-Output "[$($stageResult.status)] $($stageResult.name) ($($stageResult.durationSeconds)s)"
    }
    Write-Output "Evidence: $RunReportPath"
    if ($null -ne $RunError) {
        Write-Output "Error: $RunError"
    }
}
if ($null -ne $RunError) {
    exit 1
}

param(
    [Parameter(Mandatory=$true)][string]$TextPath,
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [int]$Rate = 0
)
$ErrorActionPreference = 'Stop'
$voice = $null
$stream = $null
try {
    $voice = New-Object -ComObject SAPI.SpVoice
    $voices = $voice.GetVoices()
    for ($i = 0; $i -lt $voices.Count; $i++) {
        # Prefer an English installed voice; do not download voice packages.
        if ($voices.Item($i).GetAttribute('Language') -match '(^|;)(409|809)(;|$)') {
            $voice.Voice = $voices.Item($i)
            break
        }
    }
    $voice.Rate = $Rate
    $stream = New-Object -ComObject SAPI.SpFileStream
    $stream.Format.Type = 22
    $stream.Open($OutputPath, 3, $false)
    $voice.AudioOutputStream = $stream
    $text = [System.IO.File]::ReadAllText($TextPath, [System.Text.Encoding]::UTF8)
    [void]$voice.Speak($text, 0)
    $stream.Close()
    Write-Output $voice.Voice.GetDescription()
} finally {
    if ($null -ne $stream) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($stream) }
    if ($null -ne $voice) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($voice) }
}
